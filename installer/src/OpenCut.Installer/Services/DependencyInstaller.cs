using System.Diagnostics;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

/// <summary>
/// Installs optional Python packages via pip using system Python.
/// </summary>
public class DependencyInstaller
{
    private static readonly string[] PythonCandidates = ["python", "python3", "py"];

    public void InstallDeps(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        var stepName = "Installing optional tools";

        if (!config.InstallOptionalDeps || config.SelectedDeps.Count == 0)
        {
            Report(progress, step, totalSteps, stepName, "Skipped (none selected).", LogLevel.Debug);
            return;
        }

        // Find system Python
        var python = FindPython();
        if (python == null)
        {
            Report(progress, step, totalSteps, stepName,
                "Python not found in PATH — optional tools skipped. Install from python.org and add to PATH.",
                LogLevel.Warning);
            return;
        }

        Report(progress, step, totalSteps, stepName, $"Using Python: {python}");

        foreach (var package in config.SelectedDeps)
        {
            Report(progress, step, totalSteps, stepName, $"Installing {package}...");
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = python,
                    Arguments = $"-m pip install {package} -q",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                if (process == null)
                {
                    Report(progress, step, totalSteps, stepName,
                        $"Failed to start pip for {package}.", LogLevel.Error);
                    continue;
                }

                // Consume both streams to prevent pipe deadlock
                var stdoutTask = process.StandardOutput.ReadToEndAsync();
                var stderrTask = process.StandardError.ReadToEndAsync();

                if (!process.WaitForExit(300000)) // 5 min per package
                {
                    try { process.Kill(entireProcessTree: true); } catch { /* best effort */ }
                    Report(progress, step, totalSteps, stepName,
                        $"Timed out installing {package} (5 min).", LogLevel.Warning);
                    continue;
                }

                var stderr = stderrTask.Result;

                if (process.ExitCode == 0)
                {
                    Report(progress, step, totalSteps, stepName,
                        $"{package} installed successfully.", LogLevel.Success);
                }
                else
                {
                    var msg = string.IsNullOrWhiteSpace(stderr) ? $"exit code {process.ExitCode}" : stderr.Trim();
                    Report(progress, step, totalSteps, stepName,
                        $"Failed to install {package}: {msg}", LogLevel.Warning);
                }
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, stepName,
                    $"Error installing {package}: {ex.Message}", LogLevel.Warning);
            }
        }
    }

    private static string? FindPython()
    {
        foreach (var name in PythonCandidates)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = name,
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                if (process == null) continue;
                process.WaitForExit(10000);

                if (process.ExitCode == 0)
                    return name;
            }
            catch
            {
                // Not found, try next
            }
        }

        return null;
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
