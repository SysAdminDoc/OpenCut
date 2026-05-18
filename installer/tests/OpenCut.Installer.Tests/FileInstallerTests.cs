using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class FileInstallerTests
{
    [Fact]
    public void CopyDirectoryCopiesNestedFilesAndReportsSuccess()
    {
        var root = TestPaths.CreateTempDirectory();
        var source = Path.Combine(root, "source");
        var destination = Path.Combine(root, "destination");
        var nested = Path.Combine(source, "nested");
        Directory.CreateDirectory(nested);
        File.WriteAllText(Path.Combine(source, "root.txt"), "root");
        File.WriteAllText(Path.Combine(nested, "child.txt"), "child");

        var progress = new RecordingProgress();
        new FileInstaller().CopyDirectory(source, destination, "Copy payload", progress, 3, 18);

        Assert.Equal("root", File.ReadAllText(Path.Combine(destination, "root.txt")));
        Assert.Equal("child", File.ReadAllText(Path.Combine(destination, "nested", "child.txt")));
        Assert.Contains(progress.Items, item => item.Level == LogLevel.Success && item.Message.Contains("Copied 2 files"));
    }

    [Fact]
    public void CopyFileCreatesDestinationDirectoryAndOverwritesExistingFile()
    {
        var root = TestPaths.CreateTempDirectory();
        var source = Path.Combine(root, "payload.txt");
        var destination = Path.Combine(root, "install", "payload.txt");
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        File.WriteAllText(source, "new");
        File.WriteAllText(destination, "old");

        var progress = new RecordingProgress();
        new FileInstaller().CopyFile(source, destination, "Copy file", progress, 4, 18);

        Assert.Equal("new", File.ReadAllText(destination));
        Assert.Contains(progress.Items, item => item.Level == LogLevel.Success && item.Message.Contains("payload.txt"));
    }

    [Fact]
    public void CopyDirectoryReportsWarningWhenSourceIsMissing()
    {
        var root = TestPaths.CreateTempDirectory();
        var progress = new RecordingProgress();

        new FileInstaller().CopyDirectory(
            Path.Combine(root, "missing"),
            Path.Combine(root, "destination"),
            "Copy payload",
            progress,
            3,
            18);

        Assert.Contains(progress.Items, item => item.Level == LogLevel.Warning && item.Message.Contains("Source not found"));
    }
}
