using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class FileInstaller
{
    /// <summary>
    /// Recursively copy a source directory to a destination, reporting progress.
    /// </summary>
    public void CopyDirectory(string source, string destination, string stepName,
        IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        if (!Directory.Exists(source))
        {
            Report(progress, step, totalSteps, stepName, $"Source not found: {source}", LogLevel.Warning);
            return;
        }

        Directory.CreateDirectory(destination);

        var files = Directory.GetFiles(source, "*", SearchOption.AllDirectories);
        int count = 0;

        foreach (var file in files)
        {
            var relativePath = Path.GetRelativePath(source, file);
            var destFile = Path.Combine(destination, relativePath);
            var destDir = Path.GetDirectoryName(destFile);

            if (destDir != null)
                Directory.CreateDirectory(destDir);

            File.Copy(file, destFile, overwrite: true);
            count++;

            if (count % 50 == 0 || count == files.Length)
            {
                Report(progress, step, totalSteps, stepName,
                    $"Copied {count}/{files.Length} files...", LogLevel.Debug);
            }
        }

        Report(progress, step, totalSteps, stepName,
            $"Copied {count} files to {destination}", LogLevel.Success);
    }

    /// <summary>
    /// Copy a single file.
    /// </summary>
    public void CopyFile(string source, string destination, string stepName,
        IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        if (!File.Exists(source))
        {
            Report(progress, step, totalSteps, stepName, $"File not found: {source}", LogLevel.Warning);
            return;
        }

        var dir = Path.GetDirectoryName(destination);
        if (dir != null) Directory.CreateDirectory(dir);

        File.Copy(source, destination, overwrite: true);
        Report(progress, step, totalSteps, stepName,
            $"Copied {Path.GetFileName(source)}", LogLevel.Success);
    }

    /// <summary>
    /// Ensure a directory exists.
    /// </summary>
    public void EnsureDirectory(string path, string stepName,
        IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Directory.CreateDirectory(path);
        Report(progress, step, totalSteps, stepName, $"Created directory: {path}", LogLevel.Success);
    }

    private static void Report(IProgress<InstallProgress> progress, int step, int total,
        string stepName, string message, LogLevel level)
    {
        progress.Report(new InstallProgress
        {
            StepNumber = step,
            TotalSteps = total,
            StepName = stepName,
            Message = message,
            Level = level
        });
    }
}
