"""F213 - Inno Setup install/uninstall smoke wiring.

The destructive installer execution itself runs only on disposable Windows CI
workers after the Inno executable is built. These cross-platform tests pin the
script and workflow contract so the smoke cannot silently disappear.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_inno_installer.ps1"
BUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yml"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_inno_smoke_script_has_profile_mutation_guard():
    text = _read(SMOKE_SCRIPT)
    assert "$env:CI" in text
    assert "AllowLocalProfileMutation" in text
    assert "~/.opencut" in text
    assert "Remove-Item -LiteralPath $installFull -Recurse -Force" in text
    assert "Assert-UnderDirectory" in text


def test_inno_smoke_script_runs_silent_install_and_uninstall():
    text = _read(SMOKE_SCRIPT)
    required = [
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/TASKS=",
        "/DIR=$installFull",
        "installer.json",
        "installer_kind",
        "unins*.exe",
        "OpenCut-Server.exe",
        "OpenCut-Launcher.vbs",
    ]
    for needle in required:
        assert needle in text


def test_windows_build_workflow_runs_inno_smoke_after_build():
    text = _read(BUILD_WORKFLOW)
    assert "Build Windows installer (Inno Setup)" in text
    assert "Smoke test Windows installer (Inno)" in text
    assert "scripts/smoke_inno_installer.ps1" in text
    assert "installer/dist/OpenCut-Setup-*.exe" in text
    assert text.index("Build Windows installer (Inno Setup)") < text.index(
        "Smoke test Windows installer (Inno)"
    )


def test_release_smoke_includes_inno_smoke_contract_tests():
    text = _read(RELEASE_SMOKE)
    assert "tests/test_inno_installer_smoke.py" in text

