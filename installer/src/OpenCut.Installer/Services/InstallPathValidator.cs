using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public enum InstallPathVerdict
{
    /// <summary>The path is usable as-is.</summary>
    Accepted,

    /// <summary>The folder contains unrelated files and must not be reused.</summary>
    RequiresDedicatedFolder,

    /// <summary>Refused outright; installing here would be destructive.</summary>
    Rejected,
}

public sealed record InstallPathResult(
    InstallPathVerdict Verdict,
    string NormalizedPath,
    string Message);

/// <summary>
/// Normalises and vets the install directory the user chose.
/// </summary>
/// <remarks>
/// The uninstaller deletes the install directory recursively and finishes with
/// <c>rmdir /s /q</c>. Without this gate a user who browsed to <c>D:\</c> or to
/// an existing media folder would lose that whole tree on uninstall, so the
/// path is normalised once, here, before it is ever stored in
/// <see cref="InstallConfig.InstallPath"/>.
/// </remarks>
public static class InstallPathValidator
{
    public const int MaxPathLength = 200;

    /// <summary>Folder name appended when the user picks a bare drive root.</summary>
    public const string AppFolderName = "OpenCut";

    /// <summary>
    /// Normalise <paramref name="rawPath"/> and decide whether installing
    /// there is safe. The returned path is always fully qualified when the
    /// verdict is not <see cref="InstallPathVerdict.Rejected"/>.
    /// </summary>
    public static InstallPathResult Validate(string? rawPath)
    {
        var trimmed = (rawPath ?? string.Empty).Trim().Trim('"');
        if (string.IsNullOrWhiteSpace(trimmed))
        {
            return new InstallPathResult(InstallPathVerdict.Rejected, string.Empty,
                "Please select an install location.");
        }

        string fullPath;
        try
        {
            fullPath = Path.GetFullPath(trimmed);
        }
        catch (Exception)
        {
            return new InstallPathResult(InstallPathVerdict.Rejected, string.Empty,
                "Install path contains invalid characters.");
        }

        if (!Path.IsPathFullyQualified(trimmed))
        {
            return new InstallPathResult(InstallPathVerdict.Rejected, string.Empty,
                "Enter a full path including the drive letter, for example C:\\Program Files\\OpenCut.");
        }

        fullPath = TrimTrailingSeparator(fullPath);
        var message = string.Empty;

        // A bare drive root is never an install target. `rmdir /s /q D:\` on
        // uninstall would wipe the drive. Put the app in its own folder.
        if (IsDriveRoot(fullPath))
        {
            fullPath = Path.Combine(fullPath, AppFolderName);
            message = $"Installing into {fullPath}. OpenCut needs its own folder, "
                    + "not the root of a drive.";
        }

        if (fullPath.Length > MaxPathLength)
        {
            return new InstallPathResult(InstallPathVerdict.Rejected, string.Empty,
                $"Install path is too long (max {MaxPathLength} characters).");
        }

        if (IsProtectedLocation(fullPath))
        {
            return new InstallPathResult(InstallPathVerdict.Rejected, string.Empty,
                "That location is used by Windows or holds your personal files. "
                + "Uninstalling OpenCut deletes the install folder, so choose a "
                + "dedicated folder such as C:\\Program Files\\OpenCut.");
        }

        if (DirectoryHasForeignContent(fullPath))
        {
            var warning = $"{fullPath} already contains files that are not part of OpenCut. "
                        + "Choose an empty folder or a dedicated OpenCut folder so uninstalling can't remove unrelated files.";
            return new InstallPathResult(InstallPathVerdict.RequiresDedicatedFolder, fullPath, warning);
        }

        return new InstallPathResult(InstallPathVerdict.Accepted, fullPath, message);
    }

    /// <summary>
    /// Last line of defence before the uninstaller deletes a directory tree.
    /// </summary>
    public static bool IsSafeToDelete(string? path)
    {
        if (string.IsNullOrWhiteSpace(path)) return false;

        var trimmed = path.Trim().Trim('"');

        // Check the caller's string, not the resolved one: `GetFullPath` would
        // silently turn "OpenCut" into a fully-qualified path under whatever
        // the current directory happens to be.
        if (!Path.IsPathFullyQualified(trimmed)) return false;

        string fullPath;
        try
        {
            fullPath = TrimTrailingSeparator(Path.GetFullPath(trimmed));
        }
        catch (Exception)
        {
            return false;
        }

        if (IsDriveRoot(fullPath)) return false;
        if (IsProtectedLocation(fullPath)) return false;

        // Refuse anything sitting directly under a drive root with no name,
        // and refuse paths that resolve to a single segment such as `C:\Users`.
        var root = Path.GetPathRoot(fullPath) ?? string.Empty;
        var relative = fullPath[root.Length..].Trim(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return relative.Length > 0;
    }

    private static string TrimTrailingSeparator(string path)
    {
        var root = Path.GetPathRoot(path) ?? string.Empty;
        if (path.Length <= root.Length) return path;

        var trimmed = path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return trimmed.Length < root.Length ? root : trimmed;
    }

    private static bool IsDriveRoot(string fullPath)
    {
        var root = Path.GetPathRoot(fullPath);
        if (string.IsNullOrEmpty(root)) return false;
        return string.Equals(
            TrimTrailingSeparator(fullPath).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Directories that must never become the install root, because the
    /// uninstaller would delete them: Windows itself, the shared roots of
    /// Program Files / ProgramData, and the user's own profile folders.
    /// </summary>
    private static bool IsProtectedLocation(string fullPath)
    {
        foreach (var folder in ProtectedFolders())
        {
            if (string.IsNullOrWhiteSpace(folder)) continue;

            string candidate;
            try { candidate = TrimTrailingSeparator(Path.GetFullPath(folder)); }
            catch (Exception) { continue; }

            if (string.Equals(fullPath, candidate, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        // The Windows tree is off-limits in its entirety, not just at its root.
        var windows = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
        return !string.IsNullOrWhiteSpace(windows) && IsUnder(fullPath, windows);
    }

    private static IEnumerable<string> ProtectedFolders()
    {
        yield return Environment.GetFolderPath(Environment.SpecialFolder.Windows);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.System);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.SystemX86);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.CommonProgramFiles);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.CommonProgramFilesX86);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.MyVideos);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        yield return Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    }

    private static bool IsUnder(string candidate, string ancestor)
    {
        string normalizedAncestor;
        try { normalizedAncestor = TrimTrailingSeparator(Path.GetFullPath(ancestor)); }
        catch (Exception) { return false; }

        if (string.Equals(candidate, normalizedAncestor, StringComparison.OrdinalIgnoreCase))
            return true;

        return candidate.StartsWith(
            normalizedAncestor + Path.DirectorySeparatorChar,
            StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// True when the directory exists, is not empty, and its contents were not
    /// produced by a previous OpenCut install.
    /// </summary>
    private static bool DirectoryHasForeignContent(string fullPath)
    {
        try
        {
            if (!Directory.Exists(fullPath)) return false;

            var entries = Directory.EnumerateFileSystemEntries(fullPath).ToList();
            if (entries.Count == 0) return false;

            return !entries.All(entry =>
            {
                var name = Path.GetFileName(entry);
                return KnownInstallEntries.Contains(name);
            });
        }
        catch (Exception)
        {
            // If the directory cannot be read, err toward asking the user.
            return true;
        }
    }

    private static readonly HashSet<string> KnownInstallEntries = new(StringComparer.OrdinalIgnoreCase)
    {
        "server",
        "ffmpeg",
        "extension",
        "logs",
        "OpenCut-Uninstall.exe",
        "OpenCut-Server.bat",
        "OpenCut-Launcher.vbs",
        "LICENSE",
        "README.md",
    };
}
