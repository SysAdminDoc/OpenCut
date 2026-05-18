"""F212 - WPF installer xUnit and headless smoke wiring."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PROJECT = (
    REPO_ROOT
    / "installer"
    / "tests"
    / "OpenCut.Installer.Tests"
    / "OpenCut.Installer.Tests.csproj"
)
UNIT_SCRIPT = REPO_ROOT / "scripts" / "test_wpf_installer.ps1"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_wpf_installer.ps1"
BUILD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yml"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"
APP = REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "App.xaml.cs"
CLI_OPTIONS = (
    REPO_ROOT
    / "installer"
    / "src"
    / "OpenCut.Installer"
    / "CommandLine"
    / "InstallerCommandLineOptions.cs"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_wpf_xunit_project_compiles_installer_contract_sources():
    text = _read(TEST_PROJECT)
    assert "net9.0-windows" in text
    assert "xunit" in text
    assert "Microsoft.NET.Test.Sdk" in text
    assert "CommandLine\\*.cs" in text
    assert "Models\\*.cs" in text
    assert "Services\\*.cs" in text
    assert "SubStream.cs" in text


def test_wpf_installer_has_quiet_command_line_entrypoint():
    app = _read(APP)
    options = _read(CLI_OPTIONS)

    assert "InstallerCommandLineOptions.Parse" in app
    assert "HeadlessInstallerRunner.Run" in app
    assert "Shutdown(HeadlessInstallerRunner.Run(options))" in app
    assert "--install" in options
    assert "--uninstall" in options
    assert "--quiet" in options
    assert "--user-data-dir" in options
    assert "--no-cep-extension" in options
    assert "--no-path-update" in options
    assert "--no-register-uninstaller" in options


def test_wpf_unit_script_runs_dotnet_test_project():
    text = _read(UNIT_SCRIPT)
    assert "OpenCut.Installer.Tests.csproj" in text
    assert "dotnet test" in text
    assert "[F212]" in text


def test_wpf_smoke_script_uses_temp_profile_and_quiet_install_uninstall():
    text = _read(SMOKE_SCRIPT)
    required = [
        "$env:CI",
        "AllowLocalProfileMutation",
        "OpenCut-WPF-Setup-*.exe",
        "--install",
        "--uninstall",
        "--quiet",
        "--user-data-dir=$userDataDir",
        "--no-cep-extension",
        "--no-path-update",
        "--no-register-uninstaller",
        "installer_kind",
        "wpf",
        "OpenCut-Uninstall.exe",
        "Assert-UnderDirectory",
    ]
    for needle in required:
        assert needle in text


def test_windows_build_workflow_runs_wpf_xunit_and_headless_smoke():
    text = _read(BUILD_WORKFLOW)
    assert "Test WPF installer contracts" in text
    assert "./scripts/test_wpf_installer.ps1" in text
    assert "Smoke test Windows installer (WPF)" in text
    assert "scripts/smoke_wpf_installer.ps1" in text
    assert "installer/dist/wpf/OpenCut-WPF-Setup-*.exe" in text
    assert text.index("Build Windows installer (WPF)") < text.index(
        "Smoke test Windows installer (WPF)"
    )
    assert text.index("Smoke test Windows installer (WPF)") < text.index(
        "Build Windows installer (Inno Setup)"
    )


def test_release_smoke_includes_wpf_installer_contract_tests():
    text = _read(RELEASE_SMOKE)
    assert "tests/test_wpf_installer_test_suite.py" in text
