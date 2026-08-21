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

    [Fact]
    public void BothUninstallersPreserveUserDataUnlessRemovalIsExplicit()
    {
        var wpfPage = ReadSource("Pages", "UninstallPage.xaml");
        Assert.Contains("OpenCut user data is preserved by default", wpfPage);
        Assert.Contains("x:Name=\"RemoveUserDataCheckBox\"", wpfPage);

        var inno = File.ReadAllText(Path.Combine(RepoRoot, "OpenCut.iss"));
        Assert.Contains("InteractiveRemoveUserData", inno);
        Assert.Contains("HasUninstallParameter('/REMOVEUSERDATA')", inno);
        Assert.Contains("BackupAndRemoveUserData(ConfigDir)", inno);
        Assert.DoesNotContain("Type: filesandordirs; Name: \"{%USERPROFILE}\\.opencut\"", inno);
    }

    [Fact]
    public void InstallerUsesTheCompactRadiusScaleWithoutTextPills()
    {
        var allowed = new HashSet<int> { 0, 4, 6, 8, 10, 12 };
        var xamlFiles = Directory.EnumerateFiles(InstallerRoot, "*.xaml", SearchOption.AllDirectories);

        foreach (var file in xamlFiles)
        {
            var source = File.ReadAllText(file);
            var matches = System.Text.RegularExpressions.Regex.Matches(
                source,
                "(?:CornerRadius=\\\"(?<value>[^\\\"]+)\\\"|Property=\\\"CornerRadius\\\"\\s+Value=\\\"(?<value>[^\\\"]+)\\\")");

            foreach (System.Text.RegularExpressions.Match match in matches)
            {
                foreach (var token in match.Groups["value"].Value.Split(','))
                {
                    Assert.True(
                        int.TryParse(token, out var radius) && allowed.Contains(radius),
                        $"Unsupported corner radius '{token}' in {Path.GetRelativePath(RepoRoot, file)}");
                }
            }
        }
    }

    [Fact]
    public void InstallerUsesInlineSafetyFeedbackInsteadOfConfirmationDialogs()
    {
        var options = ReadSource("Pages", "OptionsPage.xaml.cs");
        var uninstall = ReadSource("Pages", "UninstallPage.xaml.cs");
        var mainWindow = ReadSource("MainWindow.xaml");
        var progress = ReadSource("Pages", "ProgressPage.xaml");

        Assert.DoesNotContain("MessageBox.Show", options);
        Assert.DoesNotContain("MessageBox.Show", uninstall);
        Assert.Contains("PathValidationPanel", ReadSource("Pages", "OptionsPage.xaml"));
        Assert.DoesNotContain("Color=\"#", mainWindow);
        Assert.DoesNotContain("Foreground=\"White\"", mainWindow);
        Assert.DoesNotContain("#40FFFFFF", progress);
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
