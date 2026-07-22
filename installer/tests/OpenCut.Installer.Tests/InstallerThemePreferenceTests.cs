using OpenCut.Installer.Models;

namespace OpenCut.Installer.Tests;

public sealed class InstallerThemePreferenceTests
{
    [Theory]
    [InlineData("light", 0, InstallerTheme.Light)]
    [InlineData("LIGHT", 0, InstallerTheme.Light)]
    [InlineData("dark", 1, InstallerTheme.Dark)]
    [InlineData("Dark", 1, InstallerTheme.Dark)]
    [InlineData(null, 1, InstallerTheme.Light)]
    [InlineData(null, 0, InstallerTheme.Dark)]
    [InlineData("system", 1, InstallerTheme.Light)]
    [InlineData("invalid", "invalid", InstallerTheme.Dark)]
    public void ResolveUsesExplicitOverrideThenWindowsPreference(
        string? overrideValue,
        object? appsUseLightTheme,
        InstallerTheme expected)
    {
        Assert.Equal(
            expected,
            InstallerThemePreference.Resolve(overrideValue, appsUseLightTheme));
    }
}
