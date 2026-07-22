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
    private readonly UserDataRemovalService _userDataRemovalService = new();
    private readonly InstallerManifestCleaner _installerManifestCleaner = new();

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

        // Step 2: Preserve user data, or back it up before explicitly requested removal.
        step = 2;
        Report(progress, step, totalSteps, "Protecting user data",
            _config.RemoveUserData
                ? $"Creating a verified backup before removing {_config.UserDataPath}..."
                : $"Preserving OpenCut user data at {_config.UserDataPath}.");
        var dataResult = _userDataRemovalService.Apply(_config);
        if (dataResult is not null)
        {
            Report(progress, step, totalSteps, "Protecting user data",
                dataResult.SourceExisted
                    ? $"Backed up {dataResult.FileCount} files to {dataResult.BackupPath}; user data removed."
                    : $"No user data directory found at {_config.UserDataPath}.",
                LogLevel.Success);
        }
        else
        {
            // Preserve user-created data, but remove installation metadata that
            // would otherwise report an uninstalled copy as still present.
            _installerManifestCleaner.Remove(_config);
            Report(progress, step, totalSteps, "Protecting user data",
                "Preserved user data and removed stale installer metadata.",
                LogLevel.Success);
        }

        // Step 3: Remove CEP extension from Adobe folder
        step = 3;
        Report(progress, step, totalSteps, "Removing CEP extension", "Removing Adobe extension...");
        if (_config.InstallCepExtension)
        {
            _cepInstaller.RemoveExtension(_config.InstallPath);
            Report(progress, step, totalSteps, "Removing CEP extension",
                "CEP extension removed.", LogLevel.Success);
        }
        else
        {
            Report(progress, step, totalSteps, "Removing CEP extension", "Skipped (not selected).", LogLevel.Debug);
        }

        // Step 4: Remove FFmpeg from PATH
        step = 4;
        Report(progress, step, totalSteps, "Cleaning PATH", "Removing FFmpeg from user PATH...");
        if (_config.UpdatePath)
        {
            _registryManager.RemoveFromPath(_config.FfmpegPath);
            Report(progress, step, totalSteps, "Cleaning PATH",
                "FFmpeg removed from PATH.", LogLevel.Success);
        }
        else
        {
            Report(progress, step, totalSteps, "Cleaning PATH", "Skipped (not selected).", LogLevel.Debug);
        }

        // Step 5: Remove shortcuts
        step = 5;
        Report(progress, step, totalSteps, "Removing shortcuts", "Removing shortcuts...");
        if (_config.CreateDesktopShortcut || _config.CreateStartMenuShortcut || _config.CreateStartupShortcut)
        {
            _shortcutCreator.RemoveShortcuts();
            Report(progress, step, totalSteps, "Removing shortcuts",
                "All shortcuts removed.", LogLevel.Success);
        }
        else
        {
            Report(progress, step, totalSteps, "Removing shortcuts", "Skipped (not selected).", LogLevel.Debug);
        }

        // Step 6: Remove registry entries
        step = 6;
        Report(progress, step, totalSteps, "Cleaning registry", "Removing registry entries...");
        _registryManager.RemoveInstallKey();
        if (_config.RegisterUninstaller)
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
