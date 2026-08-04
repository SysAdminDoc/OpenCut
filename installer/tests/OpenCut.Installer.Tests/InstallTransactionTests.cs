using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public sealed class InstallTransactionTests
{
    [Fact]
    public void RollbackRestoresPreviousTreeAndRemovesPartialFiles()
    {
        var root = TestPaths.CreateTempDirectory();
        try
        {
            var installPath = Path.Combine(root, "OpenCut");
            Directory.CreateDirectory(installPath);
            File.WriteAllText(Path.Combine(installPath, "old-version.txt"), "old");

            using var transaction = new InstallTransaction(
                new InstallConfig { InstallPath = installPath },
                captureExternalState: false);
            var progress = new RecordingProgress();
            transaction.Begin(progress, totalSteps: 18);
            transaction.PrepareCleanInstallRoot();
            File.WriteAllText(Path.Combine(installPath, "partial-new-version.txt"), "new");

            transaction.Rollback(progress, step: 7, totalSteps: 18);

            Assert.Equal("old", File.ReadAllText(Path.Combine(installPath, "old-version.txt")));
            Assert.False(File.Exists(Path.Combine(installPath, "partial-new-version.txt")));
            Assert.Contains(progress.Items, item =>
                item.StepName == "Rollback complete" && item.Level == LogLevel.Success);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void CommitLeavesCleanInstallTreeWithoutThePreviousFiles()
    {
        var root = TestPaths.CreateTempDirectory();
        try
        {
            var installPath = Path.Combine(root, "OpenCut");
            Directory.CreateDirectory(installPath);
            File.WriteAllText(Path.Combine(installPath, "stale.txt"), "old");

            using var transaction = new InstallTransaction(
                new InstallConfig { InstallPath = installPath },
                captureExternalState: false);
            var progress = new RecordingProgress();
            transaction.Begin(progress, totalSteps: 18);
            transaction.PrepareCleanInstallRoot();
            File.WriteAllText(Path.Combine(installPath, "current.txt"), "new");
            transaction.Commit(progress, step: 18, totalSteps: 18);

            Assert.False(File.Exists(Path.Combine(installPath, "stale.txt")));
            Assert.Equal("new", File.ReadAllText(Path.Combine(installPath, "current.txt")));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
