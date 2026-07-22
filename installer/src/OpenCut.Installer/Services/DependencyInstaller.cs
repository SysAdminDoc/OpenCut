using System.Diagnostics;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

/// <summary>
/// Installs optional Python packages via pip using system Python.
/// </summary>
public class DependencyInstaller
{
    private static readonly string[] PythonCandidates = ["python", "python3", "py"];
    private static readonly IReadOnlyDictionary<string, string> SupportedRequirements =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["auto-editor"] = "auto-editor>=29.3,<30",
            ["edge-tts"] = "edge-tts>=6.1,<7",
            ["mediapipe"] = "mediapipe>=0.10,<1",
        };

    internal static bool TryGetSupportedRequirement(string package, out string requirement) =>
        SupportedRequirements.TryGetValue(package, out requirement!);

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
                "Python 3.11-3.14 not found in PATH — optional tools skipped. Install a supported Python from python.org.",
                LogLevel.Warning);
            return;
        }

        Report(progress, step, totalSteps, stepName, $"Using Python: {python}");

        foreach (var package in config.SelectedDeps)
        {
            if (!TryGetSupportedRequirement(package, out var requirement))
            {
                Report(progress, step, totalSteps, stepName,
                    $"Unsupported optional tool '{package}'. Choose auto-editor, edge-tts, or mediapipe.",
                    LogLevel.Error);
                continue;
            }
            Report(progress, step, totalSteps, stepName, $"Installing {package}...");
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = python,
                    Arguments = $"-m pip install \"{requirement}\" -q",
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
                    Arguments = "-c \"import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)\"",
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
