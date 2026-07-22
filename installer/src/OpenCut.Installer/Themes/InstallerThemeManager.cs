using System.Windows;
using Microsoft.Win32;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Themes;

public static class InstallerThemeManager
{
    public const string ThemeOverrideEnvironmentVariable = "OPENCUT_INSTALLER_THEME";

    private const string PersonalizeRegistryPath =
        @"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize";

    public static InstallerTheme ApplyPreferredTheme(ResourceDictionary resources)
    {
        object? appsUseLightTheme = null;
        try
        {
            appsUseLightTheme = Registry.GetValue(
                PersonalizeRegistryPath,
                "AppsUseLightTheme",
                0);
        }
        catch
        {
            // Registry access can be unavailable in restricted launch contexts.
            // The installer keeps its established dark default in that case.
        }

        var theme = InstallerThemePreference.Resolve(
            Environment.GetEnvironmentVariable(ThemeOverrideEnvironmentVariable),
            appsUseLightTheme);
        Apply(resources, theme);
        return theme;
    }

    public static void Apply(ResourceDictionary resources, InstallerTheme theme)
    {
        ArgumentNullException.ThrowIfNull(resources);

        var sourceName = theme == InstallerTheme.Light
            ? "Themes/CatppuccinLatte.xaml"
            : "Themes/CatppuccinMocha.xaml";
        var replacement = new ResourceDictionary
        {
            Source = new Uri(sourceName, UriKind.Relative)
        };

        var themeIndex = resources.MergedDictionaries
            .Select((dictionary, index) => new { dictionary, index })
            .FirstOrDefault(item =>
                item.dictionary.Source?.OriginalString.Contains(
                    "Catppuccin",
                    StringComparison.OrdinalIgnoreCase) == true)
            ?.index ?? -1;

        if (themeIndex >= 0)
            resources.MergedDictionaries[themeIndex] = replacement;
        else
            resources.MergedDictionaries.Insert(0, replacement);

        resources["OpenCutInstallerTheme"] = theme.ToString().ToLowerInvariant();
    }
}
