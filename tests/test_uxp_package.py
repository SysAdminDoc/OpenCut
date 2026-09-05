"""The UXP panel has to be packageable and placeable.

Every installer lane shipped only the CEP panel: `Install.ps1`, `OpenCut.iss`,
`install.py` and the WPF installer had zero UXP references between them, while
README advertised a UXP panel for Premiere 25.6+. With Adobe's ExtendScript
support in Premiere Pro ending in September 2026, the only installable panel
was the one on a deprecation clock.

Signed `.ccx` distribution through Creative Cloud needs an Adobe identity and a
marketplace review, which is tracked separately. What is testable here is the
package build and the developer-mode sideload path.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from opencut.core.uxp_package import (
    UXP_SOURCE_DIR,
    UXPPackageError,
    build_ccx,
    iter_package_files,
    plugin_folder_name,
    read_uxp_manifest,
    sideload_target,
    uxp_plugins_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_uxp_panel_source_exists_and_declares_what_we_need():
    manifest = read_uxp_manifest()
    assert manifest["id"] == "com.opencut.uxp"
    assert manifest["version"]
    assert manifest["main"]


def test_folder_name_follows_adobes_id_underscore_version_rule():
    """Premiere will not load a sideloaded plugin from a differently named dir."""
    manifest = {"id": "com.opencut.uxp", "version": "1.55.1", "main": "index.html"}
    assert plugin_folder_name(manifest) == "com.opencut.uxp_1.55.1"


def test_windows_sideload_path_is_appdata_uxp_plugins_external():
    target = sideload_target(
        {"id": "com.opencut.uxp", "version": "9.9.9", "main": "index.html"},
        platform="windows",
        environ={"APPDATA": r"C:\Users\Test\AppData\Roaming"},
    )
    parts = target.parts
    assert parts[-5:] == ("Adobe", "UXP", "Plugins", "External", "com.opencut.uxp_9.9.9")


def test_macos_sideload_path_is_application_support_uxp_plugins_external():
    target = sideload_target(
        {"id": "com.opencut.uxp", "version": "9.9.9", "main": "index.html"},
        platform="darwin",
        environ={"HOME": "/Users/test"},
    )
    parts = target.parts
    assert parts[-7:] == (
        "Library", "Application Support", "Adobe", "UXP", "Plugins", "External",
        "com.opencut.uxp_9.9.9",
    )


def test_manifest_version_matches_the_project_version():
    """A stale plugin version changes the folder name and orphans the install."""
    from opencut import __version__

    assert read_uxp_manifest()["version"] == __version__


def test_package_excludes_development_only_files():
    names = {arcname for _, arcname in iter_package_files()}
    assert "manifest.json" in names
    assert "index.html" in names
    assert not any(name.startswith("node_modules/") for name in names)
    assert "eslint.config.mjs" not in names


def test_built_ccx_is_a_zip_with_the_manifest_at_its_root(tmp_path):
    out = tmp_path / "OpenCut-UXP.ccx"
    result = build_ccx(out)

    assert out.is_file()
    assert result["plugin_id"] == "com.opencut.uxp"
    assert result["file_count"] > 1

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        assert "manifest.json" in names, "Premiere reads manifest.json from the archive root"
        payload = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert payload["id"] == "com.opencut.uxp"
        # The entry point must actually be in the package.
        assert payload["main"] in names
        assert archive.testzip() is None


def test_rebuilding_replaces_an_existing_package(tmp_path):
    out = tmp_path / "OpenCut-UXP.ccx"
    out.write_bytes(b"stale")
    build_ccx(out)
    with zipfile.ZipFile(out) as archive:
        assert "manifest.json" in archive.namelist()


def test_a_missing_source_directory_fails_loudly(tmp_path):
    with pytest.raises(UXPPackageError, match="source missing"):
        list(iter_package_files(tmp_path / "nope"))


def test_a_manifest_without_an_id_is_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "1.0.0", "main": "i.html"}), encoding="utf-8")
    with pytest.raises(UXPPackageError, match="'id'"):
        read_uxp_manifest(tmp_path)


def test_plugins_root_is_per_user_not_program_files():
    """A per-user path keeps the installer out of an elevation prompt."""
    root = uxp_plugins_root("windows", {"APPDATA": r"C:\Users\Test\AppData\Roaming"})
    assert "Program Files" not in str(root)


# ---------------------------------------------------------------------------
# The installer lane
# ---------------------------------------------------------------------------

def test_install_py_deploys_the_uxp_panel():
    """The regression: install.py shipped CEP only."""
    source = (REPO_ROOT / "install.py").read_text(encoding="utf-8")
    assert "install_uxp_extension" in source
    assert "com.opencut.uxp" in source
    # It has to actually be called, not merely defined.
    body = source.split("def main(")[-1]
    assert "install_uxp_extension()" in body


def test_install_py_tells_the_user_about_developer_mode():
    """A sideloaded UXP plugin does nothing until Developer Mode is on."""
    source = (REPO_ROOT / "install.py").read_text(encoding="utf-8")
    assert "Developer Mode" in source


def test_uxp_source_is_not_referenced_by_the_cep_constant():
    """Guard against the two panels being conflated in the installer."""
    source = (REPO_ROOT / "install.py").read_text(encoding="utf-8")
    assert 'CEP_EXT = "com.opencut.panel"' in source
    assert str(UXP_SOURCE_DIR.name) == "com.opencut.uxp"
