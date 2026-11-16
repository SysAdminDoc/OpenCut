using System.Diagnostics;
using System.Text;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

/// <summary>
/// Installs the optional tools by running <c>OpenCut-Server.exe --install-tools</c>.
/// The server owns the pins and puts each tool where it loads it from: the
/// pinned auto-editor build in the data folder's <c>bin</c>, and edge-tts and
/// mediapipe in its <c>packages</c> folder once it has imported them.
/// </summary>
public class DependencyInstaller
{
    /// <summary>Kept in step with <c>opencut.core.optional_tools.TOOL_NAMES</c>.</summary>
    internal static readonly string[] SupportedTools = ["auto-editor", "edge-tts", "mediapipe"];

    // Two pip installs, two import checks and a 45 MB download, with room to spare.
    private static readonly TimeSpan Timeout = TimeSpan.FromMinutes(45);

    internal static bool TryNormalizeTool(string? name, out string tool)
    {
        var trimmed = name?.Trim() ?? "";
        tool = SupportedTools.FirstOrDefault(
            supported => string.Equals(supported, trimmed, StringComparison.OrdinalIgnoreCase)) ?? "";
        return tool.Length > 0;
    }

    /// <summary>
    /// What <c>OpenCut-Server.exe --python-requirement</c> printed at build
    /// time, such as "64-bit (x64) Python 3.13", at the payload's root.
    /// </summary>
    internal const string PythonRequirementEntry = "python-requirement.txt";

    private static readonly System.Text.RegularExpressions.Regex PythonRequirementPattern =
        new(@"^[A-Za-z0-9() -]{1,40} Python \d+\.\d+$");

    /// <summary>The payload's Python requirement, or null when it's missing or malformed.</summary>
    internal static string? ParsePythonRequirement(string? text)
    {
        var line = text?.Trim();
        return line != null && PythonRequirementPattern.IsMatch(line) ? line : null;
    }

    internal static string ReadPythonRequirement() =>
        ParsePythonRequirement(PayloadExtractor.ReadEntryText(Environment.ProcessPath, PythonRequirementEntry))
        ?? "";

    private static string PythonPhrase(string requirement) =>
        requirement.Length > 0
            ? $"{requirement} from python.org"
            : "the Python version the OpenCut server runs, from python.org. The install log names it if it's missing";

    internal static string OptionsCaption(string requirement) =>
        "The OpenCut server installs these where it loads them and checks each one first. " +
        $"auto-editor is a pinned download. edge-tts and mediapipe also need {PythonPhrase(requirement)}.";

    internal static string OptionsHelpText(string requirement) =>
        $"Installs the selected tools for the OpenCut server. edge-tts and mediapipe need {PythonPhrase(requirement)}.";

    internal static IReadOnlyList<string> BuildArguments(IEnumerable<string> tools) =>
        ["--install-tools", string.Join(",", tools.Distinct(StringComparer.OrdinalIgnoreCase))];

    internal static LogLevel LevelFor(string line)
    {
        if (line.StartsWith("[ok]", StringComparison.Ordinal))
            return LogLevel.Success;
        if (line.StartsWith("[failed]", StringComparison.Ordinal))
            return LogLevel.Warning;
        return LogLevel.Info;
    }

    public void InstallDeps(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        var stepName = "Installing optional tools";

        if (!config.InstallOptionalDeps || config.SelectedDeps.Count == 0)
        {
            Report(progress, step, totalSteps, stepName, "Skipped (none selected).", LogLevel.Debug);
            return;
        }

        var tools = new List<string>();
        foreach (var name in config.SelectedDeps)
        {
            if (TryNormalizeTool(name, out var tool))
                tools.Add(tool);
            else
                Report(progress, step, totalSteps, stepName,
                    $"Unsupported optional tool '{name}'. Choose auto-editor, edge-tts or mediapipe.",
                    LogLevel.Error);
        }
        if (tools.Count == 0)
            return;

        var serverExe = Path.Combine(config.ServerPath, AppConstants.ServerExeName);
        if (!File.Exists(serverExe))
        {
            Report(progress, step, totalSteps, stepName,
                "Server executable not found, so the optional tools were skipped.", LogLevel.Warning);
            return;
        }

        Report(progress, step, totalSteps, stepName, $"Installing {string.Join(", ", tools.Distinct())}...");
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = serverExe,
                WorkingDirectory = config.ServerPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            foreach (var argument in BuildArguments(tools))
                psi.ArgumentList.Add(argument);
            psi.Environment["PYTHONIOENCODING"] = "utf-8";

            using var process = Process.Start(psi);
            if (process == null)
            {
                Report(progress, step, totalSteps, stepName,
                    "Failed to start the server to install the optional tools.", LogLevel.Error);
                return;
            }

            var stderr = new StringBuilder();
            var sawStatus = false;
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data != null)
                    lock (stderr) stderr.AppendLine(e.Data);
            };
            process.BeginErrorReadLine();

            var stdoutTask = Task.Run(() =>
            {
                string? line;
                while ((line = process.StandardOutput.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;
                    var level = LevelFor(line);
                    sawStatus |= level != LogLevel.Info;
                    Report(progress, step, totalSteps, stepName, line, level);
                }
            });

            if (!process.WaitForExit(Timeout))
            {
                try { process.Kill(entireProcessTree: true); } catch { /* best effort */ }
                Report(progress, step, totalSteps, stepName,
                    $"Optional tools timed out ({Timeout.TotalMinutes:0} min).", LogLevel.Warning);
                return;
            }
            stdoutTask.Wait(TimeSpan.FromSeconds(10));
            process.WaitForExit();

            if (process.ExitCode != 0 && !sawStatus)
            {
                string detail;
                lock (stderr) detail = stderr.ToString().Trim();
                Report(progress, step, totalSteps, stepName,
                    $"Optional tools failed (exit {process.ExitCode}): {(detail.Length > 0 ? detail : "no output")}",
                    LogLevel.Warning);
            }
        }
        catch (Exception ex)
        {
            Report(progress, step, totalSteps, stepName,
                $"Error installing optional tools: {ex.Message}", LogLevel.Warning);
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
