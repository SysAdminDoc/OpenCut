using Microsoft.Win32;
using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class DependencyInstallerTests
{
    [Fact]
    public void WindowsHelpersResolveInsideSystem32()
    {
        var paths = new[]
        {
            WindowsProcessPaths.CommandProcessor,
            WindowsProcessPaths.PowerShell,
            WindowsProcessPaths.WindowsScriptHost,
            WindowsProcessPaths.Where,
        };

        Assert.All(paths, path =>
        {
            Assert.True(Path.IsPathFullyQualified(path));
            Assert.True(File.Exists(path));
            Assert.StartsWith(
                Path.GetFullPath(Environment.SystemDirectory).TrimEnd(Path.DirectorySeparatorChar),
                path,
                StringComparison.OrdinalIgnoreCase);
        });
    }

    [Fact]
    public void ExpandableUserPathRoundTripsWithoutExpandingVariables()
    {
        // A top-level key of its own: a subkey under Software\OpenCut outlived
        // the test and stopped the uninstaller from removing the app's key.
        var keyPath = $"Software\\OpenCut.InstallerTests.{Guid.NewGuid():N}";
        try
        {
            using var key = Registry.CurrentUser.CreateSubKey(keyPath)
                ?? throw new InvalidOperationException("Could not create test registry key.");
            const string rawPath = "%USERPROFILE%\\bin";
            key.SetValue("Path", rawPath, RegistryValueKind.ExpandString);

            var snapshot = RegistryManager.ReadPathValue(key);
            Assert.Equal(rawPath, snapshot.Value);
            Assert.Equal(RegistryValueKind.ExpandString, snapshot.Kind);

            RegistryManager.WritePathValue(key, snapshot.Value, snapshot.Kind);

            Assert.Equal(
                rawPath,
                key.GetValue("Path", "", RegistryValueOptions.DoNotExpandEnvironmentNames));
            Assert.Equal(RegistryValueKind.ExpandString, key.GetValueKind("Path"));
        }
        finally
        {
            Registry.CurrentUser.DeleteSubKeyTree(keyPath, throwOnMissingSubKey: false);
        }
    }

    [Theory]
    [InlineData("auto-editor", "auto-editor")]
    [InlineData("edge-tts", "edge-tts")]
    [InlineData(" MEDIAPIPE ", "mediapipe")]
    public void OptionalToolsNormalizeToTheNamesTheServerInstalls(string name, string expected)
    {
        Assert.True(DependencyInstaller.TryNormalizeTool(name, out var tool));
        Assert.Equal(expected, tool);
    }

    [Theory]
    [InlineData("whisperx")]
    [InlineData("audiocraft")]
    [InlineData("resemble-enhance")]
    [InlineData("edge-tts; whoami")]
    [InlineData("edge-tts,mediapipe")]
    [InlineData("")]
    [InlineData(null)]
    public void UnsupportedOrInjectedToolNamesAreRejected(string? name)
    {
        Assert.False(DependencyInstaller.TryNormalizeTool(name, out _));
    }

    [Fact]
    public void TheServerGetsOneInstallToolsArgumentWithEachToolOnce()
    {
        var arguments = DependencyInstaller.BuildArguments(["edge-tts", "auto-editor", "edge-tts"]);

        Assert.Equal(["--install-tools", "edge-tts,auto-editor"], arguments);
    }

    [Theory]
    [InlineData("[ok] edge-tts is ready", LogLevel.Success)]
    [InlineData("[failed] mediapipe: no wheel", LogLevel.Warning)]
    [InlineData("Downloading auto-editor 31.8.0 (45 MB)...", LogLevel.Info)]
    [InlineData("  [ok] indented is just output", LogLevel.Info)]
    public void ServerStatusLinesSetTheLogLevel(string line, LogLevel expected)
    {
        Assert.Equal(expected, DependencyInstaller.LevelFor(line));
    }

    [Fact]
    public void AMissingServerSkipsTheToolsWithAWarning()
    {
        var config = new InstallConfig
        {
            InstallPath = Path.Combine(Path.GetTempPath(), $"OpenCut.NoServer.{Guid.NewGuid():N}"),
            InstallOptionalDeps = true,
        };
        config.SelectedDeps.Add("auto-editor");
        var reports = new List<InstallProgress>();

        new DependencyInstaller().InstallDeps(config, new ImmediateProgress(reports), 1, 1);

        var last = Assert.Single(reports, r => r.Level != LogLevel.Info);
        Assert.Equal(LogLevel.Warning, last.Level);
        Assert.Contains("Server executable not found", last.Message);
    }

    private sealed class ImmediateProgress(List<InstallProgress> reports) : IProgress<InstallProgress>
    {
        public void Report(InstallProgress value) => reports.Add(value);
    }
}
