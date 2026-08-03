"""`Install.ps1` must kill the backend listener, never a connected client.

`netstat -ano | Select-String ":5679 "` matches ESTABLISHED rows as well as
the LISTENING row, and the last column of an ESTABLISHED row is the PID of
one end of that connection. While the CEP panel is open, that end is Premiere
Pro - so the old parse terminated Premiere on every install or uninstall and
took unsaved project work with it.

The behavioural test extracts `Get-OpenCutListenerPids` from `Install.ps1`,
runs it under PowerShell against recorded `netstat -ano` output, and asserts
only the LISTENING PID comes back. It skips where no PowerShell is available;
the textual assertions below run everywhere.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "Install.ps1"

# A realistic capture: the backend listens on 5679 (PID 4242) and Premiere
# (PID 9001) plus a second client (PID 9002) hold established connections to
# it. 5680 is a listener owned by an unrelated process. The IPv6 listener row
# for 5679 belongs to the same backend.
RECORDED_NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1080
  TCP    127.0.0.1:5679         0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:5679         127.0.0.1:56011        ESTABLISHED     4242
  TCP    127.0.0.1:56011        127.0.0.1:5679         ESTABLISHED     9001
  TCP    127.0.0.1:56012        127.0.0.1:5679         ESTABLISHED     9002
  TCP    127.0.0.1:5680         0.0.0.0:0              LISTENING       7777
  TCP    [::1]:5679             [::]:0                 LISTENING       4242
  UDP    0.0.0.0:5353           *:*                                    2200
"""


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


def _run_listener_lookup(port: int, netstat_output: str) -> list[str]:
    """Run the real helper from `Install.ps1` against recorded netstat text."""
    shell = _powershell()
    assert shell is not None
    script = (
        _extract_function(_read_install_ps1(), "Get-OpenCutListenerPids")
        + "\n$lines = @'\n"
        + netstat_output
        + "\n'@ -split \"`r?`n\"\n"
        + f"ConvertTo-Json -Compress @(Get-OpenCutListenerPids -Port {port} -NetstatOutput $lines)\n"
    )
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return [str(v) for v in json.loads(completed.stdout.strip().splitlines()[-1])]


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is not available")
def test_only_the_listening_pid_is_selected_for_the_backend_port():
    selected = _run_listener_lookup(5679, RECORDED_NETSTAT)

    assert selected == ["4242"]
    assert "9001" not in selected, "Premiere's client-side PID must never be killed"
    assert "9002" not in selected
    assert "7777" not in selected, "a listener on a different port must not match"


@pytest.mark.skipif(_powershell() is None, reason="PowerShell is not available")
def test_a_port_with_only_client_connections_selects_nothing():
    clients_only = "\n".join(
        line for line in RECORDED_NETSTAT.splitlines() if "LISTENING" not in line
    )
    assert _run_listener_lookup(5679, clients_only) == []


def _executable_source() -> str:
    """`Install.ps1` with block comments and line comments removed."""
    source = re.sub(r"<#.*?#>", "", _read_install_ps1(), flags=re.DOTALL)
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def test_install_script_no_longer_takes_the_last_column_of_every_matching_row():
    source = _executable_source()
    assert 'Select-String ":5679 "' not in source
    assert 'Select-String ":$port "' not in source
    assert re.search(r"\(\$_ -split '\\s\+'\)\[-1\]", source) is None


def test_both_port_kill_paths_go_through_the_listener_helper():
    source = _read_install_ps1()
    assert source.count("Get-OpenCutListenerPids -Port") >= 2
    assert "Get-OpenCutListenerPids -Port 5679" in source
    assert "Get-OpenCutListenerPids -Port $port" in source


def test_listener_helper_prefers_typed_connection_state():
    helper = _extract_function(_read_install_ps1(), "Get-OpenCutListenerPids")
    assert "Get-NetTCPConnection" in helper
    assert "-State Listen" in helper
    # The netstat fallback must check the state column and the *local* address
    # column, not merely "a line containing the port".
    assert "$fields[3] -ne 'LISTENING'" in helper
    assert "$fields[1] -notmatch" in helper
