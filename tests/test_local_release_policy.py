"""Local release policy guards.

The repository intentionally keeps build, test, advisory, and release artifact
validation on the maintainer workstation. Active docs must describe local
commands instead of removed hosted automation.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_DOCS = (
    REPO_ROOT / "SECURITY.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "DEVELOPMENT.md",
    REPO_ROOT / "docs" / "INSTALLER_POLICY.md",
    REPO_ROOT / "docs" / "LINUX_DISTRIBUTION.md",
    REPO_ROOT / "docs" / "MACOS_NOTARIZATION.md",
    REPO_ROOT / "docs" / "WINDOWS_CODESIGNING.md",
    REPO_ROOT / "docs" / "RELEASE_PROVENANCE.md",
)

OPTIONAL_ACTIVE_LOCAL_BUILD_SURFACES = (
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "docs" / "UXP_MIGRATION.md",
)

ACTIVE_LOCAL_BUILD_SCRIPTS = (
    REPO_ROOT / "scripts" / "build_wpf_installer_ci.ps1",
    REPO_ROOT / "scripts" / "smoke_wpf_installer.ps1",
    REPO_ROOT / "scripts" / "smoke_inno_installer.ps1",
)

FORBIDDEN_ACTIVE_DOC_PATTERNS = {
    "GitHub Actions": re.compile(r"GitHub Actions"),
    "Dependabot": re.compile(r"Dependabot"),
    ".github/workflows": re.compile(r"\.github[\\/]+workflows"),
    "CI": re.compile(r"\bCI\b"),
}

FORBIDDEN_LOCAL_BUILD_PATTERNS = {
    "GitHub Actions": re.compile(r"GitHub Actions"),
    ".github/workflows": re.compile(r"\.github[\\/]+workflows"),
    "CI/CD": re.compile(r"\bCI/CD\b"),
    "$env:CI": re.compile(r"\$env:CI\b"),
    "GITHUB_OUTPUT": re.compile(r"\bGITHUB_OUTPUT\b"),
    "RUNNER_OS": re.compile(r"\bRUNNER_OS\b"),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _existing(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in paths if path.exists())


def test_active_docs_do_not_promise_removed_hosted_automation():
    offenders: list[str] = []
    for path in ACTIVE_DOCS:
        text = _read(path)
        for label, pattern in FORBIDDEN_ACTIVE_DOC_PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {label}")

    assert offenders == []


def test_active_local_build_surfaces_do_not_reference_removed_workflows():
    offenders: list[str] = []
    paths = (
        *ACTIVE_LOCAL_BUILD_SCRIPTS,
        *_existing(OPTIONAL_ACTIVE_LOCAL_BUILD_SURFACES),
    )
    for path in paths:
        text = _read(path)
        for label, pattern in FORBIDDEN_LOCAL_BUILD_PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {label}")

    assert offenders == []


def test_local_release_docs_name_the_supported_commands():
    docs = {path.name: _read(path) for path in ACTIVE_DOCS}

    assert "python scripts/release_smoke.py --json" in docs["CONTRIBUTING.md"]
    assert "python scripts/sync_version.py --check" in docs["CONTRIBUTING.md"]
    assert "dotnet publish installer/src/OpenCut.Installer/OpenCut.Installer.csproj" in docs["DEVELOPMENT.md"]
    assert "scripts/smoke_wpf_installer.ps1 -AllowLocalProfileMutation" in docs["WINDOWS_CODESIGNING.md"]
    assert "scripts/sign_windows_artifacts.ps1 -FailOnExpiringCert" in docs["WINDOWS_CODESIGNING.md"]
    assert "scripts/notarize_macos.sh --check-env" in docs["MACOS_NOTARIZATION.md"]
    assert "bash scripts/build_linux_packages.sh" in docs["LINUX_DISTRIBUTION.md"]
    assert "python scripts/verify_ffmpeg_provenance.py --manifest" in docs["RELEASE_PROVENANCE.md"]
    assert "scripts/build_wpf_installer_ci.ps1" in docs["INSTALLER_POLICY.md"]


def test_removed_automation_files_stay_absent():
    assert not (REPO_ROOT / ".github" / "dependabot.yml").exists()

    workflows = REPO_ROOT / ".github" / "workflows"
    if not workflows.exists():
        return
    assert list(workflows.glob("*")) == []


def test_release_smoke_runs_local_release_policy_guard():
    smoke = _read(REPO_ROOT / "scripts" / "release_smoke.py")

    assert "tests/test_local_release_policy.py" in smoke
