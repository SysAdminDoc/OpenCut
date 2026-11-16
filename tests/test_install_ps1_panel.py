"""Install.ps1 copies the Premiere panel without its DevTools file unless asked.

The panel's ``.debug`` file makes Premiere open a Chromium DevTools port (8842)
that any program on the machine can connect to. The README recommends
Install.bat for source installs, so that copy must leave it out too.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "Install.ps1"
_FOUND_POWERSHELL = shutil.which("powershell")
POWERSHELL = _FOUND_POWERSHELL or "powershell"

DEBUG_FILE = '<ExtensionList><Extension Id="com.opencut.panel.main"><HostList>' \
    '<Host Name="PPRO" Port="8842"/></HostList></Extension></ExtensionList>\n'


def _extract_function(source: str, name: str) -> str:
    start = source.index(f"function {name} {{")
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}")


def _copy_panel(tmp_path: Path, keep_devtools: bool) -> Path:
    source = tmp_path / "checkout" / "com.opencut.panel"
    (source / "CSXS").mkdir(parents=True)
    (source / "CSXS" / "manifest.xml").write_text("<ExtensionManifest />\n", encoding="utf-8")
    (source / ".debug").write_text(DEBUG_FILE, encoding="utf-8")
    target_parent = tmp_path / "CEP" / "extensions"
    target_parent.mkdir(parents=True)
    target = target_parent / "com.opencut.panel"

    function = _extract_function(INSTALL_PS1.read_text(encoding="utf-8"), "Copy-OpenCutPanel")
    flag = " -KeepDevTools" if keep_devtools else ""
    script = (
        "$ErrorActionPreference = 'Stop'\n" + function + "\n"
        f"Copy-OpenCutPanel -Source $env:OC_SRC -Target $env:OC_DST{flag}\n"
    )
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        env={**os.environ, "OC_SRC": str(source), "OC_DST": str(target)},
        timeout=120,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (target / "CSXS" / "manifest.xml").is_file()
    return target


windows_powershell = pytest.mark.skipif(
    sys.platform != "win32" or _FOUND_POWERSHELL is None,
    reason="Install.ps1 runs under Windows PowerShell",
)


@windows_powershell
def test_installed_panel_has_no_devtools_file(tmp_path):
    target = _copy_panel(tmp_path, keep_devtools=False)
    assert not (target / ".debug").exists()


@windows_powershell
def test_panel_devtools_switch_keeps_it(tmp_path):
    target = _copy_panel(tmp_path, keep_devtools=True)
    assert (target / ".debug").read_text(encoding="utf-8") == DEBUG_FILE


def test_every_panel_copy_goes_through_the_helper():
    source = INSTALL_PS1.read_text(encoding="utf-8")
    step = source[source.index("# Step 5: CEP Extension") : source.index("# Step 6: Create Launcher Scripts")]
    assert "Copy-Item $extensionSrc" not in step
    assert step.count("Copy-OpenCutPanel -Source $extensionSrc") == 2
    assert step.count("-KeepDevTools:$PanelDevTools") == 2
