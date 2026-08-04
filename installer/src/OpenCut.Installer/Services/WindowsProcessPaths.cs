namespace OpenCut.Installer.Services;

/// <summary>
/// Resolves Windows-provided executables without relying on the installer's
/// current directory or an elevated process's PATH.
/// </summary>
public static class WindowsProcessPaths
{
    public static string CommandProcessor => ResolveSystem32("cmd.exe");

    public static string PowerShell => ResolveSystem32(
        Path.Combine("WindowsPowerShell", "v1.0", "powershell.exe"));

    public static string WindowsScriptHost => ResolveSystem32("wscript.exe");

    public static string Where => ResolveSystem32("where.exe");

    private static string ResolveSystem32(string relativePath)
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("Windows helper resolution requires Windows.");

        var systemDirectory = Environment.SystemDirectory;
        if (!Path.IsPathFullyQualified(systemDirectory))
            throw new InvalidOperationException("Windows system directory is not an absolute path.");

        var systemRoot = Path.GetFullPath(systemDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var fullPath = Path.GetFullPath(Path.Combine(systemRoot, relativePath));
        if (!fullPath.StartsWith(systemRoot + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase) || !File.Exists(fullPath))
        {
            throw new FileNotFoundException("Required Windows helper was not found.", fullPath);
        }

        return fullPath;
    }
}
