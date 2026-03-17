using System.Diagnostics;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class UninstallEngine
{
    private readonly InstallConfig _config;
    private readonly ProcessKiller _processKiller = new();
    private readonly RegistryManager _registryManager = new();
    private readonly ShortcutCreator _shortcutCreator = new();
    private readonly CepInstaller _cepInstaller = new();

    public UninstallEngine(InstallConfig config)
    {
        _config = config;

        // If install path not set, try to detect from registry
        if (string.IsNullOrEmpty(_config.InstallPath) ||
            _config.InstallPath == AppConstants.DefaultInstallPath)
        {
            var regPath = RegistryManager.GetInstalledPath();
            if (regPath != null)
                _config.InstallPath = regPath;
        }
    }

    public void RunUninstall(IProgress<InstallProgress> progress)
    {
        int totalSteps = 8;
        int step = 0;

        // Step 1: Kill running processes
        step = 1;
        _processKiller.KillOpenCutProcesses(progress, step, totalSteps);

        // Step 2: Remove CEP extension from Adobe folder
        step = 2;
        Report(progress, step, totalSteps, "Removing CEP extension", "Removing Adobe extension...");
        _cepInstaller.RemoveExtension(_config.InstallPath);
        Report(progress, step, totalSteps, "Removing CEP extension",
            "CEP extension removed.", LogLevel.Success);

        // Step 3: Remove FFmpeg from PATH
        step = 3;
        Report(progress, step, totalSteps, "Cleaning PATH", "Removing FFmpeg from user PATH...");
        _registryManager.RemoveFromPath(_config.FfmpegPath);
        Report(progress, step, totalSteps, "Cleaning PATH",
            "FFmpeg removed from PATH.", LogLevel.Success);

        // Step 4: Remove shortcuts
        step = 4;
        Report(progress, step, totalSteps, "Removing shortcuts", "Removing shortcuts...");
        _shortcutCreator.RemoveShortcuts();
        Report(progress, step, totalSteps, "Removing shortcuts",
            "All shortcuts removed.", LogLevel.Success);

        // Step 5: Remove config directory (~/.opencut)
        step = 5;
        Report(progress, step, totalSteps, "Removing config", "Removing user config...");
        var configDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".opencut");
        if (Directory.Exists(configDir))
        {
            try
            {
                Directory.Delete(configDir, recursive: true);
                Report(progress, step, totalSteps, "Removing config",
                    "Config directory removed.", LogLevel.Success);
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, "Removing config",
                    $"Could not remove config: {ex.Message}", LogLevel.Warning);
            }
        }
        else
        {
            Report(progress, step, totalSteps, "Removing config",
                "No config directory found.", LogLevel.Debug);
        }

        // Step 6: Remove registry entries
        step = 6;
        Report(progress, step, totalSteps, "Cleaning registry", "Removing registry entries...");
        _registryManager.RemoveInstallKey();
        _registryManager.RemoveUninstallEntry();
        Report(progress, step, totalSteps, "Cleaning registry",
            "Registry entries removed.", LogLevel.Success);

        // Step 7: Delete install directory (skip uninstaller exe which is running)
        step = 7;
        Report(progress, step, totalSteps, "Removing files", "Deleting install directory...");
        if (Directory.Exists(_config.InstallPath))
        {
            try
            {
                var selfPath = Environment.ProcessPath;
                var files = Directory.GetFiles(_config.InstallPath, "*", SearchOption.AllDirectories);
                int deleted = 0;

                foreach (var file in files)
                {
                    // Skip the running uninstaller exe
                    if (selfPath != null && file.Equals(selfPath, StringComparison.OrdinalIgnoreCase))
                        continue;

                    try { File.Delete(file); deleted++; }
                    catch { /* File in use */ }
                }

                // Try to remove empty subdirectories
                foreach (var dir in Directory.GetDirectories(_config.InstallPath, "*", SearchOption.AllDirectories)
                    .OrderByDescending(d => d.Length))
                {
                    try { Directory.Delete(dir); }
                    catch { /* Not empty or in use */ }
                }

                Report(progress, step, totalSteps, "Removing files",
                    $"Deleted {deleted} files.", LogLevel.Success);
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, "Removing files",
                    $"Partial cleanup: {ex.Message}", LogLevel.Warning);
            }
        }

        // Step 8: Broadcast environment change
        step = 8;
        Report(progress, step, totalSteps, "Finalizing", "Broadcasting environment changes...");
        NativeMethods.BroadcastEnvironmentChange();
        Report(progress, totalSteps, totalSteps, "Uninstall complete",
            "OpenCut has been removed.", LogLevel.Success);
    }

    /// <summary>
    /// Schedule self-delete via cmd.exe timeout after the process exits.
    /// </summary>
    public void ScheduleSelfDelete()
    {
        try
        {
            var selfPath = Environment.ProcessPath;
            if (selfPath == null || !File.Exists(selfPath)) return;

            var installDir = _config.InstallPath;

            // Use cmd.exe to wait, then delete the exe and parent directory
            var cmd = $"/c timeout /t 3 /nobreak >nul & del /f /q \"{selfPath}\" & rmdir /s /q \"{installDir}\"";

            Process.Start(new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = cmd,
                WindowStyle = ProcessWindowStyle.Hidden,
                CreateNoWindow = true,
                UseShellExecute = false
            });
        }
        catch { /* Best effort */ }
    }

    private static void Report(IProgress<InstallProgress> progress, int step, int total,
        string stepName, string message, LogLevel level = LogLevel.Info)
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
