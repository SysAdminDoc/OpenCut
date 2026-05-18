using OpenCut.Installer.Models;

namespace OpenCut.Installer.Tests;

public class InstallConfigTests
{
    [Fact]
    public void DerivedInstallPathsStayUnderInstallRoot()
    {
        var config = new InstallConfig { InstallPath = @"C:\OpenCut Test" };

        Assert.Equal(@"C:\OpenCut Test\server", config.ServerPath);
        Assert.Equal(@"C:\OpenCut Test\ffmpeg", config.FfmpegPath);
        Assert.Equal(@"C:\OpenCut Test\extension\com.opencut.panel", config.ExtensionPath);
        Assert.Equal(@"C:\OpenCut Test\logs", config.LogsPath);
        Assert.Equal(@"C:\OpenCut Test\OpenCut-Uninstall.exe", config.UninstallExePath);
    }

    [Fact]
    public void InstallerManifestPathUsesConfigurableUserDataRoot()
    {
        var config = new InstallConfig { UserDataPath = @"C:\Temp\OpenCutProfile" };

        Assert.Equal(@"C:\Temp\OpenCutProfile\installer.json", config.InstallerManifestPath);
    }

    [Fact]
    public void ProgressPercentHandlesZeroTotals()
    {
        Assert.Equal(0, new InstallProgress { StepNumber = 5, TotalSteps = 0 }.OverallPercent);
        Assert.Equal(50, new InstallProgress { StepNumber = 4, TotalSteps = 8 }.OverallPercent);
    }
}
