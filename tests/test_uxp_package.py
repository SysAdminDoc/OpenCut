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


# ---------------------------------------------------------------------------
# Every lane must agree on the folder name, and clear superseded versions
# ---------------------------------------------------------------------------

def _lane_sources() -> dict[str, str]:
    return {
        "Install.ps1": (REPO_ROOT / "Install.ps1").read_text(encoding="utf-8", errors="replace"),
        "OpenCut.iss": (REPO_ROOT / "OpenCut.iss").read_text(encoding="utf-8", errors="replace"),
        "install.py": (REPO_ROOT / "install.py").read_text(encoding="utf-8", errors="replace"),
        "CepInstaller.cs": (
            REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Services" / "CepInstaller.cs"
        ).read_text(encoding="utf-8", errors="replace"),
    }


def test_every_lane_places_the_uxp_panel():
    """Three of the four lanes shipped CEP only."""
    for name, source in _lane_sources().items():
        assert "com.opencut.uxp" in source, f"{name} does not place the UXP panel"


def test_every_lane_targets_the_external_plugins_directory():
    """Either the lane names the path, or it delegates to the one that does."""
    for name, source in _lane_sources().items():
        names_path = "UXP" in source and "External" in source
        delegates = "sideload_target" in source
        assert names_path or delegates, (
            f"{name} does not target the UXP/Plugins/External directory Premiere reads"
        )


def test_every_lane_clears_superseded_versions_on_install():
    """The folder name carries the version, so an upgrade orphans the old one.

    Premiere in Developer Mode loads every plugin under External, so leaving
    com.opencut.uxp_<old> beside the new one means two copies of the panel.
    """
    sources = _lane_sources()
    # Each lane must match by prefix somewhere, not only delete the exact
    # current folder name.
    assert 'Filter "$uxpId`_*"' in sources["Install.ps1"] or 'com.opencut.uxp_*' in sources["Install.ps1"]
    assert "RemoveUXPVersions" in sources["OpenCut.iss"]
    assert 'UxpExtensionId + "_*"' in sources["CepInstaller.cs"]


def test_the_windows_lanes_remove_every_version_on_uninstall():
    sources = _lane_sources()
    # Install.ps1's uninstall branch and the WPF RemoveExtension both prefix-match.
    assert sources["Install.ps1"].count("com.opencut.uxp_*") >= 1
    assert "RemoveUXPVersions" in sources["OpenCut.iss"]
    assert 'UxpExtensionId + "_*"' in sources["CepInstaller.cs"]


def test_the_inno_uninstall_is_not_pinned_to_one_version_alone():
    """[UninstallDelete] names one version; code must sweep the rest."""
    source = (REPO_ROOT / "OpenCut.iss").read_text(encoding="utf-8", errors="replace")
    uninstall_block = source.split("procedure CurUninstallStepChanged", 1)[1][:2000]
    assert "RemoveUXPVersions" in uninstall_block, (
        "the Inno uninstall only removes the version it was built with"
    )


def test_the_folder_name_matches_across_every_lane():
    """Four lanes derive <id>_<version> four ways; they must agree.

    Install.ps1 and install.py read manifest.json; OpenCut.iss uses
    MyAppVersion and the WPF installer uses AppConstants.AppVersion. They line
    up only because sync_version.py rewrites all of them, and nothing checked.
    """
    import re

    manifest = read_uxp_manifest()
    expected = plugin_folder_name(manifest)

    iss = (REPO_ROOT / "OpenCut.iss").read_text(encoding="utf-8", errors="replace")
    iss_version = re.search(r'#define MyAppVersion "([^"]+)"', iss)
    assert iss_version, "OpenCut.iss no longer defines MyAppVersion"
    assert f"{manifest['id']}_{iss_version.group(1)}" == expected, (
        "OpenCut.iss would create a folder Premiere does not look for"
    )

    constants = (
        REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Models" / "AppConstants.cs"
    ).read_text(encoding="utf-8", errors="replace")
    cs_version = re.search(r'AppVersion\s*=\s*"([^"]+)"', constants)
    assert cs_version, "AppConstants no longer defines AppVersion"
    assert f"{manifest['id']}_{cs_version.group(1)}" == expected, (
        "the Windows installer would create a folder Premiere does not look for"
    )
