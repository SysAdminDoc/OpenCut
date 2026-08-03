"""CSXS PlayerDebugMode coverage must agree across every install path.

Adobe CEP only loads an unsigned panel when
``HKCU\\Software\\Adobe\\CSXS.<n>\\PlayerDebugMode`` is set for the CSXS
revision the host uses. Premiere CC 2023+ moved to CSXS 13 and current
builds are on 18, so an install path that stops at CSXS 12 silently
produces a Premiere in which the panel never appears — while still
reporting success.

Four install paths write that key (the WPF installer, ``Install.ps1``,
``OpenCut.iss``, and ``install.py``). This module pins them to one set so
they cannot drift apart again.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

APP_CONSTANTS = (
    REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Models" / "AppConstants.cs"
)
REGISTRY_MANAGER = (
    REPO_ROOT
    / "installer"
    / "src"
    / "OpenCut.Installer"
    / "Services"
    / "RegistryManager.cs"
)
INSTALL_PS1 = REPO_ROOT / "Install.ps1"
INNO_SCRIPT = REPO_ROOT / "OpenCut.iss"
INSTALL_PY = REPO_ROOT / "install.py"

# CSXS 7 (CC 2014) through 18 (Premiere 2025+).
EXPECTED_VERSIONS = frozenset(range(7, 19))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _wpf_versions() -> frozenset[int]:
    match = re.search(
        r"CsxsVersions\s*=\s*\{([^}]*)\}",
        _read(APP_CONSTANTS),
    )
    assert match, "CsxsVersions array not found in AppConstants.cs"
    return frozenset(int(v) for v in re.findall(r"\d+", match.group(1)))


def _powershell_versions() -> frozenset[int]:
    return frozenset(
        int(v)
        for v in re.findall(
            r"HKCU:\\Software\\Adobe\\CSXS\.(\d+)", _read(INSTALL_PS1)
        )
    )


def _inno_versions() -> frozenset[int]:
    source = _read(INNO_SCRIPT)
    return frozenset(
        int(v)
        for v in re.findall(
            r"HKCU\\Software\\Adobe\\CSXS\.(\d+)\"\"\s+/v PlayerDebugMode", source
        )
    )


def _install_py_versions() -> frozenset[int]:
    match = re.search(
        r"versions\s*=\s*\[str\(v\) for v in range\((\d+),\s*(\d+)\)\]",
        _read(INSTALL_PY),
    )
    assert match, "CSXS version range not found in install.py"
    return frozenset(range(int(match.group(1)), int(match.group(2))))


def test_wpf_installer_covers_modern_premiere_csxs_versions():
    assert _wpf_versions() == EXPECTED_VERSIONS


def test_powershell_installer_covers_modern_premiere_csxs_versions():
    assert _powershell_versions() == EXPECTED_VERSIONS


def test_inno_installer_covers_modern_premiere_csxs_versions():
    assert _inno_versions() == EXPECTED_VERSIONS


def test_dev_installer_covers_modern_premiere_csxs_versions():
    assert _install_py_versions() == EXPECTED_VERSIONS


def test_every_install_path_writes_the_same_csxs_set():
    per_path = {
        "AppConstants.cs": _wpf_versions(),
        "Install.ps1": _powershell_versions(),
        "OpenCut.iss": _inno_versions(),
        "install.py": _install_py_versions(),
    }
    distinct = {frozenset(v) for v in per_path.values()}
    assert len(distinct) == 1, f"CSXS coverage diverged between install paths: {per_path}"


def test_registry_manager_does_not_hardcode_a_stale_version_range():
    source = _read(REGISTRY_MANAGER)
    assert "CSXS 7-12" not in source
    # The log strings must be derived from the shared array, not retyped.
    assert source.count("AppConstants.CsxsVersionRange") >= 2
