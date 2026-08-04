using Microsoft.Win32;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class DependencyInstallerTests
{
    [Fact]
    public void SupportedPythonResolutionReturnsAnAbsoluteExecutable()
    {
        var path = PythonResolver.FindSupportedPath()
            ?? throw new InvalidOperationException("A supported Python interpreter is required for this installer test.");

        Assert.True(Path.IsPathFullyQualified(path));
        Assert.True(File.Exists(path));
        Assert.Equal(".exe", Path.GetExtension(path), ignoreCase: true);
    }

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
        var keyPath = $"Software\\OpenCut\\InstallerTests\\{Guid.NewGuid():N}";
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
    [InlineData("auto-editor", "auto-editor>=29.3,<30")]
    [InlineData("edge-tts", "edge-tts>=6.1,<7")]
    [InlineData("mediapipe", "mediapipe>=0.10,<1")]
    [InlineData("MEDIAPIPE", "mediapipe>=0.10,<1")]
    public void OptionalToolsResolveOnlyToAuditedRequirements(string package, string expected)
    {
        Assert.True(DependencyInstaller.TryGetSupportedRequirement(package, out var requirement));
        Assert.Equal(expected, requirement);
    }

    [Theory]
    [InlineData("whisperx")]
    [InlineData("audiocraft")]
    [InlineData("resemble-enhance")]
    [InlineData("edge-tts; whoami")]
    public void UnsupportedOrInjectedPackageNamesAreRejected(string package)
    {
        Assert.False(DependencyInstaller.TryGetSupportedRequirement(package, out _));
    }
}
