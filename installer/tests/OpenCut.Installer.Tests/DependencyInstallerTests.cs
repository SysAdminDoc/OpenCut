using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class DependencyInstallerTests
{
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
