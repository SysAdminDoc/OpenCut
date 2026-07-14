using OpenCut.Installer.CommandLine;

namespace OpenCut.Installer.Tests;

public class CommandLineOptionsTests
{
    [Fact]
    public void ParseQuietInstallAppliesHeadlessSmokeFlags()
    {
        var options = InstallerCommandLineOptions.Parse(
        [
            "--install",
            "--quiet",
            "--dir=C:\\Temp\\OpenCut Smoke",
            "--user-data-dir=C:\\Temp\\OpenCutProfile",
            "--log=C:\\Temp\\install.log",
            "--no-desktop-shortcut",
            "--no-start-menu-shortcut",
            "--no-startup-shortcut",
            "--no-cep-extension",
            "--no-player-debug-mode",
            "--no-path-update",
            "--no-register-uninstaller",
            "--no-whisper-model",
            "--no-optional-deps"
        ]);

        Assert.Equal(InstallerCommandMode.QuietInstall, options.Mode);
        Assert.True(options.IsQuiet);
        Assert.Equal("C:\\Temp\\OpenCut Smoke", options.Config.InstallPath);
        Assert.Equal("C:\\Temp\\OpenCutProfile", options.Config.UserDataPath);
        Assert.Equal("C:\\Temp\\install.log", options.LogPath);
        Assert.False(options.Config.CreateDesktopShortcut);
        Assert.False(options.Config.CreateStartMenuShortcut);
        Assert.False(options.Config.CreateStartupShortcut);
        Assert.False(options.Config.InstallCepExtension);
        Assert.False(options.Config.SetPlayerDebugMode);
        Assert.False(options.Config.UpdatePath);
        Assert.False(options.Config.RegisterUninstaller);
        Assert.False(options.Config.RemoveUserData);
        Assert.False(options.Config.DownloadWhisperModel);
        Assert.False(options.Config.InstallOptionalDeps);
    }

    [Fact]
    public void ParseQuietUninstallKeepsSameSafetyFlags()
    {
        var options = InstallerCommandLineOptions.Parse(
        [
            "/uninstall",
            "/silent",
            "/dir=C:\\Temp\\OpenCut",
            "/user-data-dir=C:\\Temp\\OpenCutProfile",
            "/no-cep-extension",
            "/no-path-update",
            "/no-register-uninstaller"
        ]);

        Assert.Equal(InstallerCommandMode.QuietUninstall, options.Mode);
        Assert.Equal("C:\\Temp\\OpenCut", options.Config.InstallPath);
        Assert.Equal("C:\\Temp\\OpenCutProfile", options.Config.UserDataPath);
        Assert.False(options.Config.InstallCepExtension);
        Assert.False(options.Config.UpdatePath);
        Assert.False(options.Config.RegisterUninstaller);
    }

    [Fact]
    public void ParseInteractiveUninstallWithoutQuietKeepsGuiMode()
    {
        var options = InstallerCommandLineOptions.Parse(["--uninstall"]);

        Assert.Equal(InstallerCommandMode.InteractiveUninstall, options.Mode);
        Assert.False(options.IsQuiet);
        Assert.False(options.Config.RemoveUserData);
    }

    [Fact]
    public void ParseQuietUninstallRequiresDedicatedDataRemovalFlag()
    {
        var options = InstallerCommandLineOptions.Parse(
        [
            "--uninstall",
            "--quiet",
            "--remove-user-data",
            "--user-data-dir=C:\\Temp\\OpenCutProfile",
            "--user-data-backup-dir=C:\\Temp\\OpenCutBackups"
        ]);

        Assert.Equal(InstallerCommandMode.QuietUninstall, options.Mode);
        Assert.True(options.Config.RemoveUserData);
        Assert.Equal("C:\\Temp\\OpenCutProfile", options.Config.UserDataPath);
        Assert.Equal("C:\\Temp\\OpenCutBackups", options.Config.UserDataBackupDirectory);
    }

    [Fact]
    public void ParseRejectsDataRemovalDuringInstall()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            InstallerCommandLineOptions.Parse(["--install", "--remove-user-data"]));

        Assert.Contains("valid only with --uninstall", ex.Message);
    }

    [Fact]
    public void ParseRejectsMutuallyExclusiveInstallModes()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            InstallerCommandLineOptions.Parse(["--install", "--uninstall", "--quiet"]));

        Assert.Contains("either --install or --uninstall", ex.Message);
    }
}
