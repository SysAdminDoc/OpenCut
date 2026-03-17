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
