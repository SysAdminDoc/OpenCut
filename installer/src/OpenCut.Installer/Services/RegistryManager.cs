using Microsoft.Win32;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class RegistryManager
{
    public void AddToPath(string directory, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Report(progress, step, totalSteps, "Updating PATH", $"Adding {directory} to user PATH...");

        try
        {
            using var envKey = Registry.CurrentUser.OpenSubKey(AppConstants.EnvironmentRegKey, writable: true);
            if (envKey == null) return;

            var currentPath = envKey.GetValue("Path", "") as string ?? "";

            if (currentPath.Contains(directory, StringComparison.OrdinalIgnoreCase))
            {
                Report(progress, step, totalSteps, "Updating PATH", "Already in PATH.", LogLevel.Debug);
                return;
            }

            var newPath = string.IsNullOrEmpty(currentPath)
                ? directory
                : $"{currentPath};{directory}";

            envKey.SetValue("Path", newPath, RegistryValueKind.ExpandString);

            NativeMethods.BroadcastEnvironmentChange();

            Report(progress, step, totalSteps, "Updating PATH", "FFmpeg added to user PATH.", LogLevel.Success);
        }
        catch (Exception ex)
        {
            Report(progress, step, totalSteps, "Updating PATH", $"Failed to update PATH: {ex.Message}", LogLevel.Error);
        }
    }

    public void RemoveFromPath(string directory)
    {
        try
        {
            using var envKey = Registry.CurrentUser.OpenSubKey(AppConstants.EnvironmentRegKey, writable: true);
            if (envKey == null) return;

            var currentPath = envKey.GetValue("Path", "") as string ?? "";
            var parts = currentPath.Split(';', StringSplitOptions.RemoveEmptyEntries)
                .Where(p => !p.Equals(directory, StringComparison.OrdinalIgnoreCase))
                .ToArray();

            envKey.SetValue("Path", string.Join(';', parts), RegistryValueKind.ExpandString);
            NativeMethods.BroadcastEnvironmentChange();
        }
        catch { /* Best effort */ }
    }

    public void SetPlayerDebugMode(IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Report(progress, step, totalSteps, "Setting PlayerDebugMode",
            "Enabling unsigned CEP extensions for CSXS 7-12...");

        foreach (var version in AppConstants.CsxsVersions)
        {
            try
            {
                var keyPath = $@"Software\Adobe\CSXS.{version}";
                using var key = Registry.CurrentUser.CreateSubKey(keyPath);
                key?.SetValue("PlayerDebugMode", "1", RegistryValueKind.String);
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, "Setting PlayerDebugMode",
                    $"Failed for CSXS.{version}: {ex.Message}", LogLevel.Warning);
            }
        }

        Report(progress, step, totalSteps, "Setting PlayerDebugMode",
            "PlayerDebugMode set for CSXS 7-12.", LogLevel.Success);
    }

    public void RemovePlayerDebugMode()
    {
        // We don't remove PlayerDebugMode on uninstall since other extensions may need it
    }

    public void WriteInstallPath(string installPath, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Report(progress, step, totalSteps, "Writing registry", "Writing install path to registry...");

        try
        {
            using var key = Registry.CurrentUser.CreateSubKey(AppConstants.AppRegKey);
            key?.SetValue("InstallPath", installPath, RegistryValueKind.String);
            Report(progress, step, totalSteps, "Writing registry",
                "Install path saved to HKCU\\Software\\OpenCut.", LogLevel.Success);
        }
        catch (Exception ex)
        {
            Report(progress, step, totalSteps, "Writing registry",
                $"Failed to write registry: {ex.Message}", LogLevel.Error);
        }
    }

    public void RegisterUninstall(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Report(progress, step, totalSteps, "Registering uninstaller", "Adding to Add/Remove Programs...");

        try
        {
            using var key = Registry.LocalMachine.CreateSubKey(AppConstants.UninstallRegKey);
            if (key == null)
            {
                Report(progress, step, totalSteps, "Registering uninstaller",
                    "Failed to create registry key (requires admin).", LogLevel.Error);
                return;
            }

            key.SetValue("DisplayName", $"{AppConstants.AppName} {AppConstants.AppVersion}");
            key.SetValue("DisplayVersion", AppConstants.AppVersion);
            key.SetValue("Publisher", AppConstants.AppPublisher);
            key.SetValue("URLInfoAbout", AppConstants.AppUrl);
            key.SetValue("UninstallString", $"\"{config.UninstallExePath}\" --uninstall");
            key.SetValue("QuietUninstallString", $"\"{config.UninstallExePath}\" --uninstall --quiet");
            key.SetValue("InstallLocation", config.InstallPath);
            key.SetValue("NoModify", 1, RegistryValueKind.DWord);
            key.SetValue("NoRepair", 1, RegistryValueKind.DWord);

            var iconPath = Path.Combine(config.InstallPath, "logo.ico");
            if (File.Exists(iconPath))
                key.SetValue("DisplayIcon", iconPath);

            // Estimate installed size in KB
            try
            {
                var size = GetDirectorySize(config.InstallPath) / 1024;
                key.SetValue("EstimatedSize", (int)size, RegistryValueKind.DWord);
            }
            catch { /* Best effort */ }

            Report(progress, step, totalSteps, "Registering uninstaller",
                "Registered in Add/Remove Programs.", LogLevel.Success);
        }
        catch (Exception ex)
        {
            Report(progress, step, totalSteps, "Registering uninstaller",
                $"Failed to register: {ex.Message}", LogLevel.Error);
        }
    }

    public void RemoveUninstallEntry()
    {
        try
        {
            Registry.LocalMachine.DeleteSubKey(AppConstants.UninstallRegKey, throwOnMissingSubKey: false);
        }
        catch { /* Best effort */ }
    }

    public void RemoveInstallKey()
    {
        try
        {
            Registry.CurrentUser.DeleteSubKey(AppConstants.AppRegKey, throwOnMissingSubKey: false);
        }
        catch { /* Best effort */ }
    }

    public static string? GetInstalledPath()
    {
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(AppConstants.AppRegKey);
            return key?.GetValue("InstallPath") as string;
        }
        catch { return null; }
    }

    private static long GetDirectorySize(string path)
    {
        if (!Directory.Exists(path)) return 0;
        return Directory.GetFiles(path, "*", SearchOption.AllDirectories)
            .Sum(f => new FileInfo(f).Length);
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
