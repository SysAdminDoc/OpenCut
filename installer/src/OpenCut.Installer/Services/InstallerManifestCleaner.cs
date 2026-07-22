using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public sealed class InstallerManifestCleaner
{
    public void Remove(InstallConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        var userDataPath = Path.GetFullPath(config.UserDataPath)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var manifestPath = Path.GetFullPath(config.InstallerManifestPath);
        var expectedPrefix = userDataPath + Path.DirectorySeparatorChar;
        if (!manifestPath.StartsWith(expectedPrefix, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Installer manifest must stay inside the configured user-data directory.");

        if (File.Exists(manifestPath))
            File.Delete(manifestPath);
    }
}
