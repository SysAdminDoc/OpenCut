"""F213 - Inno Setup install/uninstall smoke wiring.

The destructive installer execution itself runs only on disposable Windows
profiles after the Inno executable is built. These cross-platform tests pin the
script and release-smoke contract so the smoke cannot silently disappear.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_inno_installer.ps1"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_inno_smoke_script_has_profile_mutation_guard():
    text = _read(SMOKE_SCRIPT)
    assert "$env:OPENCUT_INSTALLER_SMOKE" in text
    assert "AllowLocalProfileMutation" in text
    assert "OpenCut-Inno-Profile-" in text
    assert "/USERDATADIR=$userDataDir" in text
    assert 'Join-Path $env:USERPROFILE ".opencut' not in text
    assert "Remove-Item -LiteralPath $installFull -Recurse -Force" in text
    assert "Assert-UnderDirectory" in text
    assert "$env:CI" not in text


def test_inno_smoke_script_runs_silent_install_and_uninstall():
    text = _read(SMOKE_SCRIPT)
    required = [
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/TASKS=",
        "/DIR=$installFull",
        "/USERDATADIR=$userDataDir",
        "installer.json",
        "installer_kind",
        "unins*.exe",
        "OpenCut-Server.exe",
        "OpenCut-Launcher.vbs",
    ]
    for needle in required:
        assert needle in text


def test_release_smoke_includes_inno_smoke_contract_tests():
    text = _read(RELEASE_SMOKE)
    assert "tests/test_inno_installer_smoke.py" in text
