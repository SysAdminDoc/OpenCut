"""Contracts for the source installer's cleanup and interpreter selection."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "Install.ps1"


def _read_install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def _extract_function(source: str, name: str) -> str:
    start = source.index(f"function {name} {{")
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}")


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _without_comments(source: str) -> str:
    source = re.sub(r"<#.*?#>", "", source, flags=re.DOTALL)
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def test_all_pip_commands_use_the_resolved_interpreter():
    source = _without_comments(_read_install_ps1())
    direct_pip = re.findall(
        r"(?m)^\s*(?:pip|&\s+pip)\s+(?:install|uninstall|list)\b", source
    )
    assert direct_pip == []

    pip_lines = [
        line for line in source.splitlines()
        if re.search(r"\bpip\s+(?:install|uninstall|list)\b", line)
    ]
    assert pip_lines
    assert all("& $pythonCmd -m pip" in line for line in pip_lines if not line.lstrip().startswith("Write-"))


def test_uninstall_resolves_python_before_running_and_cleans_created_artifacts():
    source = _read_install_ps1()
    uninstall_start = source.index("if ($Uninstall) {")
    banner_start = source.index("# Banner", uninstall_start)
    uninstall = source[uninstall_start:banner_start]

    assert source.index("$pythonCmd = Resolve-OpenCutPython") < uninstall_start
    for artifact in (
        "Start-OpenCut.bat",
        "Start-OpenCut-Hidden.vbs",
        "Start OpenCut.lnk",
    ):
        assert artifact in uninstall
    assert "Remove-ItemProperty" in uninstall
    assert '"PlayerDebugMode"' in uninstall
    assert "Remove-OpenCutArtifact" in uninstall
    assert "Close Premiere Pro" in uninstall
    assert "exit $script:ExitCode" in uninstall


def test_launcher_templates_quote_the_absolute_python_path():
    source = _read_install_ps1()
    assert "$quotedPythonCmd = '\"{0}\"' -f $pythonCmd" in source
    assert "$quotedPythonCmd -m opencut.server" in source
    assert "$vbsCommand = ('\"{0}\" -m opencut.server' -f $pythonCmd)" in source
    assert 'WshShell.Run "$vbsCommand", 0, False' in source


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is not available")
def test_resolver_returns_an_absolute_supported_interpreter():
    shell = _powershell()
    assert shell is not None
    function = _extract_function(_read_install_ps1(), "Resolve-OpenCutPython")
    script = function + "\nConvertTo-Json -Compress @(Resolve-OpenCutPython)\n"
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    resolved = json.loads(completed.stdout.strip().splitlines()[-1])
    if isinstance(resolved, list):
        assert len(resolved) == 1
        resolved = resolved[0]
    assert Path(resolved).is_absolute()
    assert Path(resolved).suffix.lower() == ".exe"


def _parse_errors(shell: str) -> subprocess.CompletedProcess[str]:
    escaped_path = str(INSTALL_PS1).replace("'", "''")
    script = f"""
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile('{escaped_path}', [ref]$tokens, [ref]$errors) | Out-Null
if ($errors.Count -gt 0) {{ $errors | ForEach-Object {{ Write-Output $_.Message }}; exit 1 }}
"""
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is not available")
def test_install_script_parses_as_powershell():
    shell = _powershell()
    assert shell is not None
    completed = _parse_errors(shell)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell 5.1 ships with Windows only")
def test_install_script_parses_under_windows_powershell():
    """Install.bat and Uninstall.bat run ``powershell``, which is 5.1."""
    shell = shutil.which("powershell")
    assert shell is not None
    completed = _parse_errors(shell)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_install_script_bytes_read_the_same_in_windows_powershell():
    """Windows PowerShell reads a BOM-less script in the ANSI code page. A
    UTF-8 em dash then decodes to a curly quote that ends the string early,
    which broke Install.bat in 1.56.0. Keep the script ASCII or give it a BOM."""
    data = INSTALL_PS1.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return
    offenders = [
        number
        for number, line in enumerate(data.split(b"\n"), 1)
        if any(byte > 0x7E for byte in line)
    ]
    assert not offenders, f"non-ASCII bytes on Install.ps1 lines {offenders}"
