"""F212 - WPF installer xUnit and headless smoke wiring."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PROJECT = (
    REPO_ROOT
    / "installer"
    / "tests"
    / "OpenCut.Installer.Tests"
    / "OpenCut.Installer.Tests.csproj"
)
APP_PROJECT = (
    REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "OpenCut.Installer.csproj"
)
PRODUCTION_MANIFEST = APP_PROJECT.parent / "Properties" / "app.manifest"
SMOKE_MANIFEST = APP_PROJECT.parent / "Properties" / "app.smoke.manifest"
BUILDER = REPO_ROOT / "installer" / "InstallerBuilder.ps1"
INSTALLER_SOURCE = REPO_ROOT / "installer" / "src" / "OpenCut.Installer"
MOCHA_THEME = INSTALLER_SOURCE / "Themes" / "CatppuccinMocha.xaml"
LATTE_THEME = INSTALLER_SOURCE / "Themes" / "CatppuccinLatte.xaml"
THEME_MANAGER = INSTALLER_SOURCE / "Themes" / "InstallerThemeManager.cs"
UNIT_SCRIPT = REPO_ROOT / "scripts" / "test_wpf_installer.ps1"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_wpf_installer.ps1"
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
    assert "net10.0-windows" in text
    assert "xunit" in text
    assert "Microsoft.NET.Test.Sdk" in text
    assert "CommandLine\\*.cs" in text
    assert "Models\\*.cs" in text
    assert "Services\\*.cs" in text
    assert "SubStream.cs" in text


def test_wpf_projects_and_builder_require_dotnet_10():
    assert "net10.0-windows" in _read(APP_PROJECT)
    assert "net9.0" not in _read(APP_PROJECT)
    assert "net9.0" not in _read(TEST_PROJECT)
    builder = _read(BUILDER)
    assert "10.0.100" in builder
    assert "Install the .NET 10 SDK" in builder


def test_wpf_smoke_manifest_does_not_weaken_production_uac_contract():
    project = _read(APP_PROJECT)
    assert "InstallerSmokeBuild" in project
    assert "app.smoke.manifest" in project
    assert 'level="requireAdministrator"' in _read(PRODUCTION_MANIFEST)
    assert 'level="asInvoker"' in _read(SMOKE_MANIFEST)


def test_wpf_light_and_dark_themes_share_complete_resource_contract():
    key_name = "{http://schemas.microsoft.com/winfx/2006/xaml}Key"

    def resource_keys(path: Path) -> set[str]:
        return {
            value
            for element in ET.parse(path).getroot().iter()
            if (value := element.attrib.get(key_name))
        }

    mocha_keys = resource_keys(MOCHA_THEME)
    latte_keys = resource_keys(LATTE_THEME)
    assert mocha_keys == latte_keys
    for key in (
        "Base",
        "TextBrush",
        "WindowChromeBrush",
        "WindowTitleBarBrush",
        "HeroBadgeBrush",
        "SuccessTintBrush",
    ):
        assert key in mocha_keys

    manager = _read(THEME_MANAGER)
    assert "AppsUseLightTheme" in manager
    assert "OPENCUT_INSTALLER_THEME" in manager
    assert "CatppuccinLatte.xaml" in manager
    assert "CatppuccinMocha.xaml" in manager


def test_wpf_shell_chrome_uses_theme_resources_instead_of_dark_literals():
    shell_sources = "\n".join(
        _read(path)
        for path in (
            INSTALLER_SOURCE / "MainWindow.xaml",
            INSTALLER_SOURCE / "Pages" / "ProgressPage.xaml",
            INSTALLER_SOURCE / "Pages" / "WelcomePage.xaml",
        )
    )
    for dark_literal in ("#0f121a", "#20293b", "#22304a", "#20362b"):
        assert dark_literal not in shell_sources
    for resource in (
        "{DynamicResource WindowTitleBarBrush}",
        "{DynamicResource BuildBadgeBrush}",
        "{DynamicResource HeroBadgeBrush}",
        "{DynamicResource SuccessTintBrush}",
    ):
        assert resource in shell_sources
    welcome = _read(INSTALLER_SOURCE / "Pages" / "WelcomePage.xaml")
    assert 'VerticalScrollBarVisibility="Auto"' in welcome
    assert 'HorizontalScrollBarVisibility="Disabled"' in welcome
    shell_code = _read(INSTALLER_SOURCE / "MainWindow.xaml.cs")
    assert "FrameworkElement.WidthProperty" in shell_code
    assert "FrameworkElement.HeightProperty" in shell_code
    assert "PageFrame.ActualHeight" in shell_code


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
        "$env:OPENCUT_INSTALLER_SMOKE",
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
    assert "$env:CI" not in text
    assert "RUNNER_OS" not in text


def test_release_smoke_includes_wpf_installer_contract_tests():
    text = _read(RELEASE_SMOKE)
    assert "tests/test_wpf_installer_test_suite.py" in text
