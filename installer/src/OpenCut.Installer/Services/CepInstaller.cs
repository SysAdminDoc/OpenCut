using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class CepInstaller
{
    private readonly FileInstaller _fileInstaller = new();

    public void InstallExtension(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        var stepName = "Installing CEP extension";
        var source = config.ExtensionPath;
        var target = config.CepTargetPath;

        Report(progress, step, totalSteps, stepName, $"Copying extension to {target}...");

        // Ensure parent directories exist
        var parentDir = Path.GetDirectoryName(target);
        if (parentDir != null)
            Directory.CreateDirectory(parentDir);

        // Remove existing extension if present
        if (Directory.Exists(target))
        {
            try
            {
                Directory.Delete(target, recursive: true);
                Report(progress, step, totalSteps, stepName, "Removed existing extension.", LogLevel.Debug);
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, stepName,
                    $"Warning: Could not remove old extension: {ex.Message}", LogLevel.Warning);
            }
        }

        _fileInstaller.CopyDirectory(source, target, stepName, progress, step, totalSteps);

        Report(progress, step, totalSteps, stepName,
            "CEP extension installed to Adobe extensions folder.", LogLevel.Success);
    }

    /// <summary>
    /// Place the UXP panel where Premiere 25.6+ looks for a sideloaded plugin.
    /// Premiere ignores it until the user enables Settings &gt; Plugins &gt;
    /// Developer Mode, which is a preference no installer can set.
    /// </summary>
    public void InstallUxpExtension(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        var stepName = "Installing UXP panel";
        var source = config.UxpExtensionPath;
        var target = config.UxpTargetPath;

        if (!Directory.Exists(source))
        {
            Report(progress, step, totalSteps, stepName,
                "UXP panel source not found; skipping.", LogLevel.Warning);
            return;
        }

        Report(progress, step, totalSteps, stepName, $"Copying UXP panel to {target}...");

        var parentDir = Path.GetDirectoryName(target);
        if (parentDir != null)
            Directory.CreateDirectory(parentDir);

        if (Directory.Exists(target))
        {
            try
            {
                Directory.Delete(target, recursive: true);
                Report(progress, step, totalSteps, stepName, "Removed existing UXP panel.", LogLevel.Debug);
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, stepName,
                    $"Warning: Could not remove old UXP panel: {ex.Message}", LogLevel.Warning);
            }
        }

        _fileInstaller.CopyDirectory(source, target, stepName, progress, step, totalSteps);

        Report(progress, step, totalSteps, stepName,
            "UXP panel installed. Enable Settings > Plugins > Developer Mode in Premiere 25.6+, then restart it.",
            LogLevel.Success);
    }

    public void RemoveExtension(string? installPath = null)
    {
        // Remove from Adobe CEP extensions folder
        var cepPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "Adobe", "CEP", "extensions", AppConstants.CepExtensionId);

        if (Directory.Exists(cepPath))
        {
            try { Directory.Delete(cepPath, recursive: true); }
            catch { /* Best effort */ }
        }

        // Remove every installed UXP panel version. The folder carries the
        // plugin version, so match by prefix rather than assuming this build's.
        var uxpRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "Adobe", "UXP", "Plugins", "External");

        if (Directory.Exists(uxpRoot))
        {
            foreach (var dir in Directory.GetDirectories(uxpRoot, AppConstants.UxpExtensionId + "_*"))
            {
                try { Directory.Delete(dir, recursive: true); }
                catch { /* Best effort */ }
            }
        }
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
