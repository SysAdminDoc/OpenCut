namespace OpenCut.Installer.Models;

public enum InstallerTheme
{
    Dark,
    Light
}

public static class InstallerThemePreference
{
    public static InstallerTheme Resolve(string? overrideValue, object? appsUseLightTheme)
    {
        if (string.Equals(overrideValue, "light", StringComparison.OrdinalIgnoreCase))
            return InstallerTheme.Light;
        if (string.Equals(overrideValue, "dark", StringComparison.OrdinalIgnoreCase))
            return InstallerTheme.Dark;

        try
        {
            return Convert.ToInt32(appsUseLightTheme) == 1
                ? InstallerTheme.Light
                : InstallerTheme.Dark;
        }
        catch (Exception) when (appsUseLightTheme is not null)
        {
            return InstallerTheme.Dark;
        }
    }
}
