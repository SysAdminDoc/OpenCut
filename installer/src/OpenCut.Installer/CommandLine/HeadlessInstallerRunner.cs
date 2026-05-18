using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.CommandLine;

public static class HeadlessInstallerRunner
{
    public static int Run(InstallerCommandLineOptions options)
    {
        var logPath = string.IsNullOrWhiteSpace(options.LogPath)
            ? Path.Combine(Path.GetTempPath(), "OpenCut-Setup-quiet.log")
            : options.LogPath;

        var progress = new QuietInstallProgress(logPath);
        progress.WriteLine($"[{DateTime.UtcNow:O}] OpenCut installer quiet mode: {options.Mode}");

        try
        {
            if (options.Mode == InstallerCommandMode.QuietUninstall)
                new UninstallEngine(options.Config).RunUninstall(progress);
            else
                new InstallEngine(options.Config).RunInstall(progress);

            progress.WriteLine($"[{DateTime.UtcNow:O}] Completed successfully.");
            return 0;
        }
        catch (Exception ex)
        {
            progress.WriteLine($"[{DateTime.UtcNow:O}] Failed: {ex}");
            return 1;
        }
    }

    private sealed class QuietInstallProgress : IProgress<InstallProgress>
    {
        private readonly string _logPath;

        public QuietInstallProgress(string logPath)
        {
            _logPath = logPath;
            var dir = Path.GetDirectoryName(_logPath);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);
        }

        public void Report(InstallProgress value)
        {
            WriteLine(
                $"[{DateTime.UtcNow:O}] [{value.Level}] " +
                $"{value.StepNumber}/{value.TotalSteps} {value.StepName}: {value.Message}");
        }

        public void WriteLine(string message)
        {
            File.AppendAllText(_logPath, message + Environment.NewLine);
        }
    }
}
