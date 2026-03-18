using System.Diagnostics;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class WhisperDownloader
{
    public void DownloadModel(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        var stepName = "Downloading Whisper model";
        var serverExe = Path.Combine(config.ServerPath, AppConstants.ServerExeName);

        if (!File.Exists(serverExe))
        {
            Report(progress, step, totalSteps, stepName,
                "Server executable not found, skipping model download.", LogLevel.Warning);
            return;
        }

        Report(progress, step, totalSteps, stepName,
            $"Downloading Whisper model: {config.WhisperModel}...");
        Report(progress, step, totalSteps, stepName,
            "This may take several minutes depending on your connection.", LogLevel.Debug);

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = serverExe,
                Arguments = $"--download-models {config.WhisperModel}",
                WorkingDirectory = config.ServerPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process == null)
            {
                Report(progress, step, totalSteps, stepName,
                    "Failed to start server process.", LogLevel.Error);
                return;
            }

            // Consume stderr async to prevent pipe deadlock
            var stderrTask = process.StandardError.ReadToEndAsync();

            // Read stdout for progress reporting
            while (!process.StandardOutput.EndOfStream)
            {
                var line = process.StandardOutput.ReadLine();
                if (!string.IsNullOrWhiteSpace(line))
                {
                    Report(progress, step, totalSteps, stepName, line, LogLevel.Debug);
                }
            }

            if (!process.WaitForExit(600000)) // 10 minute timeout
            {
                try { process.Kill(entireProcessTree: true); } catch { /* best effort */ }
                Report(progress, step, totalSteps, stepName,
                    "Model download timed out (10 min).", LogLevel.Error);
                return;
            }

            if (process.ExitCode == 0)
            {
                Report(progress, step, totalSteps, stepName,
                    $"Whisper model '{config.WhisperModel}' downloaded successfully.", LogLevel.Success);
            }
            else
            {
                var stderr = stderrTask.Result;
                Report(progress, step, totalSteps, stepName,
                    $"Model download failed (exit {process.ExitCode}): {stderr}", LogLevel.Error);
            }
        }
        catch (Exception ex)
        {
            Report(progress, step, totalSteps, stepName,
                $"Model download error: {ex.Message}", LogLevel.Error);
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
