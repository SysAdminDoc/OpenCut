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

            var pathValue = ReadPathValue(envKey);
            var currentPath = pathValue.Value;

            var pathSegments = currentPath.Split(';', StringSplitOptions.RemoveEmptyEntries);
            if (pathSegments.Any(p => p.Equals(directory, StringComparison.OrdinalIgnoreCase)))
            {
                Report(progress, step, totalSteps, "Updating PATH", "Already in PATH.", LogLevel.Debug);
                return;
            }

            var newPath = string.IsNullOrEmpty(currentPath)
                ? directory
                : $"{currentPath};{directory}";

            WritePathValue(envKey, newPath, pathValue.Kind);

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

            var pathValue = ReadPathValue(envKey);
            var currentPath = pathValue.Value;
            var parts = currentPath.Split(';', StringSplitOptions.RemoveEmptyEntries)
                .Where(p => !p.Equals(directory, StringComparison.OrdinalIgnoreCase))
                .ToArray();

            WritePathValue(envKey, string.Join(';', parts), pathValue.Kind);
            NativeMethods.BroadcastEnvironmentChange();
        }
        catch { /* Best effort */ }
    }

    public void SetPlayerDebugMode(IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Report(progress, step, totalSteps, "Setting PlayerDebugMode",
            $"Enabling unsigned CEP extensions for CSXS {AppConstants.CsxsVersionRange}...");

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
            $"PlayerDebugMode set for CSXS {AppConstants.CsxsVersionRange}.", LogLevel.Success);
    }

    public void RemovePlayerDebugMode()
    {
        // We don't remove PlayerDebugMode on uninstall since other extensions may need it
    }

    public static (string Value, RegistryValueKind Kind) ReadPathValue(RegistryKey key)
    {
        ArgumentNullException.ThrowIfNull(key);

        var value = key.GetValue(
            "Path", "", RegistryValueOptions.DoNotExpandEnvironmentNames) as string ?? "";
        var kind = RegistryValueKind.ExpandString;
        try
        {
            kind = key.GetValueKind("Path");
        }
        catch
        {
            // A missing Path defaults to an expandable value when first written.
        }

        if (kind is not RegistryValueKind.String and not RegistryValueKind.ExpandString)
            kind = RegistryValueKind.ExpandString;
        return (value, kind);
    }

    public static void WritePathValue(RegistryKey key, string value, RegistryValueKind kind)
    {
        ArgumentNullException.ThrowIfNull(key);
        var safeKind = kind is RegistryValueKind.String or RegistryValueKind.ExpandString
            ? kind
            : RegistryValueKind.ExpandString;
        key.SetValue("Path", value, safeKind);
    }

    public RegistryInstallState CaptureInstallState()
    {
        var playerDebugModes = new Dictionary<int, RegistryValueSnapshot?>();
        foreach (var version in AppConstants.CsxsVersions)
        {
            using var key = Registry.CurrentUser.OpenSubKey($@"Software\Adobe\CSXS.{version}");
            playerDebugModes[version] = CaptureValue(key, "PlayerDebugMode");
        }

        return new RegistryInstallState
        {
            UserEnvironment = CaptureKey(Registry.CurrentUser, AppConstants.EnvironmentRegKey),
            UserApplication = CaptureKey(Registry.CurrentUser, AppConstants.AppRegKey),
            MachineUninstall = CaptureKey(Registry.LocalMachine, AppConstants.UninstallRegKey),
            PlayerDebugModes = playerDebugModes,
        };
    }

    public void RestoreInstallState(RegistryInstallState state)
    {
        ArgumentNullException.ThrowIfNull(state);

        RestoreKey(Registry.CurrentUser, AppConstants.EnvironmentRegKey, state.UserEnvironment);
        RestoreKey(Registry.CurrentUser, AppConstants.AppRegKey, state.UserApplication);
        RestoreKey(Registry.LocalMachine, AppConstants.UninstallRegKey, state.MachineUninstall);

        foreach (var version in AppConstants.CsxsVersions)
        {
            var keyPath = $@"Software\Adobe\CSXS.{version}";
            using var key = Registry.CurrentUser.OpenSubKey(keyPath, writable: true);
            var snapshot = state.PlayerDebugModes.TryGetValue(version, out var value)
                ? value
                : null;

            if (snapshot is null)
            {
                key?.DeleteValue("PlayerDebugMode", throwOnMissingValue: false);
            }
            else
            {
                using var writableKey = key ?? Registry.CurrentUser.CreateSubKey(keyPath);
                writableKey?.SetValue("PlayerDebugMode", CloneValue(snapshot.Value), snapshot.Kind);
            }
        }
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

    private static RegistryKeySnapshot CaptureKey(RegistryKey root, string path)
    {
        using var key = root.OpenSubKey(path);
        if (key is null)
            return new RegistryKeySnapshot(false, new Dictionary<string, RegistryValueSnapshot>());

        var values = new Dictionary<string, RegistryValueSnapshot>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in key.GetValueNames())
        {
            var value = CaptureValue(key, name);
            if (value is not null)
                values[name] = value;
        }

        return new RegistryKeySnapshot(true, values);
    }

    private static RegistryValueSnapshot? CaptureValue(RegistryKey? key, string name)
    {
        if (key is null)
            return null;

        var value = key.GetValue(name, null, RegistryValueOptions.DoNotExpandEnvironmentNames);
        if (value is null)
            return null;
        return new RegistryValueSnapshot(CloneValue(value), key.GetValueKind(name));
    }

    private static void RestoreKey(RegistryKey root, string path, RegistryKeySnapshot snapshot)
    {
        if (!snapshot.Exists)
        {
            root.DeleteSubKeyTree(path, throwOnMissingSubKey: false);
            return;
        }

        using var key = root.CreateSubKey(path);
        if (key is null)
            throw new InvalidOperationException($"Could not open registry key {path} for restore.");

        foreach (var currentName in key.GetValueNames())
            key.DeleteValue(currentName, throwOnMissingValue: false);
        foreach (var value in snapshot.Values)
            key.SetValue(value.Key, CloneValue(value.Value.Value), value.Value.Kind);
    }

    private static object CloneValue(object value) => value switch
    {
        string[] strings => strings.ToArray(),
        byte[] bytes => bytes.ToArray(),
        int[] ints => ints.ToArray(),
        _ => value,
    };

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
