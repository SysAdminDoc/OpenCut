"""RA-12 - UXP Hybrid plugin package validator guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.core.uxp_hybrid_package import (
    REQUIRED_MARKETPLACE_ARCHITECTURES,
    validate_uxp_hybrid_package,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_UXP = REPO_ROOT / "extension" / "com.opencut.uxp"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _write_manifest(root: Path, **overrides: object) -> None:
    manifest = {
        "manifestVersion": 6,
        "id": "com.opencut.hybrid",
        "name": "OpenCut Hybrid",
        "version": "1.0.0",
        "main": "index.html",
        "host": {"app": "premierepro", "minVersion": "25.6.0"},
        "entrypoints": [{"type": "panel", "id": "panel", "label": {"default": "OpenCut"}}],
        "addon": {"name": "opencut.uxpaddon"},
        "requiredPermissions": {"enableAddon": True},
    }
    manifest.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "index.html").write_text("<div>OpenCut</div>", encoding="utf-8")


def _write_addon(root: Path, arch: str, *, layout: str = "addons", name: str = "opencut.uxpaddon") -> Path:
    target = root / layout / Path(arch) / name if layout else root / Path(arch) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"native-binary-placeholder")
    return target


def test_live_uxp_manifest_is_valid_non_hybrid_package():
    report = validate_uxp_hybrid_package(LIVE_UXP)

    assert report.valid is True
    assert report.hybrid is False
    assert report.errors == []


def test_marketplace_hybrid_package_requires_all_architectures(tmp_path):
    _write_manifest(tmp_path)
    for arch in REQUIRED_MARKETPLACE_ARCHITECTURES:
        _write_addon(tmp_path, arch)

    report = validate_uxp_hybrid_package(tmp_path)

    assert report.valid is True
    assert report.hybrid is True
    assert report.layout_root == "addons"
    assert set(report.architectures) == set(REQUIRED_MARKETPLACE_ARCHITECTURES)


def test_hybrid_package_rejects_missing_enable_addon_permission(tmp_path):
    _write_manifest(tmp_path, requiredPermissions={})
    for arch in REQUIRED_MARKETPLACE_ARCHITECTURES:
        _write_addon(tmp_path, arch)

    report = validate_uxp_hybrid_package(tmp_path)

    assert report.valid is False
    assert any("enableAddon" in error for error in report.errors)


def test_marketplace_hybrid_package_rejects_missing_architecture(tmp_path):
    _write_manifest(tmp_path)
    _write_addon(tmp_path, "mac/arm64")
    _write_addon(tmp_path, "win/x64")

    report = validate_uxp_hybrid_package(tmp_path)

    assert report.valid is False
    assert any("mac/x64" in error for error in report.errors)


def test_independent_hybrid_package_allows_partial_architecture_with_warning(tmp_path):
    _write_manifest(tmp_path)
    _write_addon(tmp_path, "win/x64")

    report = validate_uxp_hybrid_package(tmp_path, require_marketplace_architectures=False)

    assert report.valid is True
    assert report.architectures == {"win/x64": "addons/win/x64/opencut.uxpaddon"}
    assert any("partial hybrid architecture set" in warning for warning in report.warnings)


def test_hybrid_package_rejects_addon_path_traversal(tmp_path):
    _write_manifest(tmp_path, addon={"name": "../opencut.uxpaddon"})

    report = validate_uxp_hybrid_package(tmp_path)

    assert report.valid is False
    assert any("addon.name must be a filename" in error for error in report.errors)


def test_cli_reports_live_non_hybrid_manifest_as_valid():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.validate_uxp_hybrid_package", str(LIVE_UXP), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["hybrid"] is False


def test_ra12_release_smoke_runs_hybrid_package_guardrail():
    source = RELEASE_SMOKE.read_text(encoding="utf-8", errors="replace")

    assert '"tests/test_uxp_hybrid_package.py"' in source
