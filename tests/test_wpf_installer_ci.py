"""F201 WPF installer automation wrapper tests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_wpf_installer_ci.ps1"
POLICY_DOC = REPO_ROOT / "docs" / "INSTALLER_POLICY.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_wpf_installer_ci_script_exists_and_wraps_builder():
    text = _read(SCRIPT)

    assert "F201" in text
    assert "installer\\InstallerBuilder.ps1" in text
    assert "dist\\OpenCut-Server" in text
    assert "OpenCut-WPF-Setup-" in text
    assert "GITHUB_OUTPUT" in text


def test_wpf_installer_ci_script_stages_ffmpeg_from_path():
    text = _read(SCRIPT)

    assert 'Copy-ToolFromPath "ffmpeg"' in text
    assert 'Copy-ToolFromPath "ffprobe"' in text
    assert "Get-Command \"$ToolName.exe\"" in text
    assert "ffmpeg" in text


def test_installer_policy_marks_f201_local_wrapper_done():
    text = _read(POLICY_DOC)

    assert "F201 status" in text
    assert "scripts/build_wpf_installer_ci.ps1" in text
    assert "local build wrapper" in text
    assert "DONE" in text
