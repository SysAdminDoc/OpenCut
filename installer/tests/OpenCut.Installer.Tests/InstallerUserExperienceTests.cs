namespace OpenCut.Installer.Tests;

public class InstallerUserExperienceTests
{
    private static readonly string RepoRoot = FindRepoRoot();
    private static readonly string InstallerRoot = Path.Combine(RepoRoot, "installer", "src", "OpenCut.Installer");

    [Fact]
    public void VersionBadgesBindToCurrentAppVersion()
    {
        var xamlFiles = Directory.EnumerateFiles(InstallerRoot, "*.xaml", SearchOption.AllDirectories);

        foreach (var file in xamlFiles)
        {
            Assert.DoesNotContain("v1.19.0", File.ReadAllText(file));
        }

        Assert.Contains("AppConstants.AppVersion", ReadSource("MainWindow.xaml"));
        Assert.Contains("AppConstants.AppVersion", ReadSource("Pages", "WelcomePage.xaml"));
        Assert.Contains("AppConstants.AppVersion", ReadSource("Pages", "CompletePage.xaml"));
    }

    [Fact]
    public void IconOnlyWindowControlsHaveAccessibleNames()
    {
        var mainWindow = ReadSource("MainWindow.xaml");

        Assert.Contains("AutomationProperties.Name=\"Minimize setup window\"", mainWindow);
        Assert.Contains("AutomationProperties.HelpText=\"Minimizes the OpenCut Setup window.\"", mainWindow);
        Assert.Contains("AutomationProperties.Name=\"Close setup window\"", mainWindow);
        Assert.Contains("AutomationProperties.HelpText=\"Closes the OpenCut Setup window.\"", mainWindow);
    }

    [Fact]
    public void ProgressAndLogSurfacesHaveAccessibleNames()
    {
        Assert.Contains("AutomationProperties.Name=\"Activity log\"", ReadSource("Controls", "LogPanel.xaml"));
        Assert.Contains("AutomationProperties.Name=\"Install progress\"", ReadSource("Pages", "ProgressPage.xaml"));
        Assert.Contains("AutomationProperties.Name=\"Install activity log\"", ReadSource("Pages", "ProgressPage.xaml"));
        Assert.Contains("AutomationProperties.Name=\"Uninstall progress\"", ReadSource("Pages", "UninstallPage.xaml"));
        Assert.Contains("AutomationProperties.Name=\"Uninstall activity log\"", ReadSource("Pages", "UninstallPage.xaml"));
    }

    private static string ReadSource(params string[] parts)
    {
        var pathParts = new string[parts.Length + 1];
        pathParts[0] = InstallerRoot;
        Array.Copy(parts, 0, pathParts, 1, parts.Length);
        return File.ReadAllText(Path.Combine(pathParts));
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "installer", "src", "OpenCut.Installer")))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate OpenCut repository root.");
    }
}
