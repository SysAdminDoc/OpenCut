using OpenCut.Installer.Models;

namespace OpenCut.Installer.Tests;

internal sealed class RecordingProgress : IProgress<InstallProgress>
{
    public List<InstallProgress> Items { get; } = [];

    public void Report(InstallProgress value) => Items.Add(value);
}

internal static class TestPaths
{
    public static string CreateTempDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), "OpenCut-Installer-Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }
}
