using System.IO.Compression;
using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public sealed class UserDataRemovalServiceTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(),
        $"OpenCut-Uninstall-Tests-{Guid.NewGuid():N}");

    [Fact]
    public void PreservesCustomUserDataByDefault()
    {
        var source = CreateUserData();
        var config = CreateConfig(source);

        var result = new UserDataRemovalService().Apply(config);

        Assert.Null(result);
        Assert.True(File.Exists(Path.Combine(source, "settings.json")));
    }

    [Fact]
    public void ExplicitRemovalCreatesValidatedBackupBeforeDeletingCustomPath()
    {
        var source = CreateUserData();
        var config = CreateConfig(source);
        config.RemoveUserData = true;

        var result = new UserDataRemovalService().Apply(config);

        Assert.NotNull(result);
        Assert.True(result.SourceExisted);
        Assert.Equal(2, result.FileCount);
        Assert.False(Directory.Exists(source));
        Assert.NotNull(result.BackupPath);
        Assert.True(File.Exists(result.BackupPath));
        using var archive = ZipFile.OpenRead(result.BackupPath!);
        Assert.Equal("{\"theme\":\"dark\"}", ReadEntry(archive, "settings.json"));
        Assert.Equal("pending", ReadEntry(archive, "jobs/queue.txt"));
        Assert.Contains(archive.Entries, entry =>
            entry.FullName.StartsWith("__opencut_uninstall__/", StringComparison.Ordinal));
    }

    [Fact]
    public void BackupInsideDeletionRootFailsWithoutDeletingSource()
    {
        var source = CreateUserData();
        var config = CreateConfig(source);
        config.RemoveUserData = true;
        config.UserDataBackupDirectory = Path.Combine(source, "backups");

        var error = Assert.Throws<InvalidOperationException>(() =>
            new UserDataRemovalService().Apply(config));

        Assert.Contains("outside", error.Message);
        Assert.True(File.Exists(Path.Combine(source, "settings.json")));
    }

    [Fact]
    public void LockedFileFailsClosedAndLeavesSourceUntouched()
    {
        var source = CreateUserData();
        var lockedPath = Path.Combine(source, "settings.json");
        var config = CreateConfig(source);
        config.RemoveUserData = true;

        using var locked = new FileStream(
            lockedPath, FileMode.Open, FileAccess.ReadWrite, FileShare.None);

        Assert.ThrowsAny<IOException>(() => new UserDataRemovalService().Apply(config));
        Assert.True(Directory.Exists(source));
        Assert.True(File.Exists(lockedPath));
        Assert.Empty(Directory.EnumerateFiles(config.UserDataBackupDirectory, "*.partial"));
    }

    [Fact]
    public void FilesystemRootIsNeverAcceptedAsUserData()
    {
        var source = CreateUserData();
        var config = CreateConfig(source);
        config.RemoveUserData = true;
        config.UserDataPath = Path.GetPathRoot(source)!;

        Assert.Throws<InvalidOperationException>(() =>
            new UserDataRemovalService().Apply(config));
        Assert.True(Directory.Exists(source));
    }

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    private string CreateUserData()
    {
        var source = Path.Combine(_root, "custom-profile");
        Directory.CreateDirectory(Path.Combine(source, "jobs"));
        File.WriteAllText(Path.Combine(source, "settings.json"), "{\"theme\":\"dark\"}");
        File.WriteAllText(Path.Combine(source, "jobs", "queue.txt"), "pending");
        return source;
    }

    private InstallConfig CreateConfig(string source) => new()
    {
        InstallPath = Path.Combine(_root, "app"),
        UserDataPath = source,
        UserDataBackupDirectory = Path.Combine(_root, "backups")
    };

    private static string ReadEntry(ZipArchive archive, string name)
    {
        var entry = archive.GetEntry(name) ?? throw new InvalidDataException(name);
        using var reader = new StreamReader(entry.Open());
        return reader.ReadToEnd();
    }
}
