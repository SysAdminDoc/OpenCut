using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public sealed class InstallerManifestCleanerTests
{
    [Fact]
    public void RemoveDeletesOnlyInstallerMetadata()
    {
        var temp = TestPaths.CreateTempDirectory();
        try
        {
            var settingsPath = Path.Combine(temp, "settings.json");
            var manifestPath = Path.Combine(temp, AppConstants.InstallerManifestFile);
            File.WriteAllText(settingsPath, "{}");
            File.WriteAllText(manifestPath, "{}");

            new InstallerManifestCleaner().Remove(new InstallConfig
            {
                UserDataPath = temp
            });

            Assert.False(File.Exists(manifestPath));
            Assert.True(File.Exists(settingsPath));
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public void RemoveIsIdempotentWhenManifestIsAbsent()
    {
        var temp = TestPaths.CreateTempDirectory();
        try
        {
            new InstallerManifestCleaner().Remove(new InstallConfig
            {
                UserDataPath = temp
            });
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }
}
