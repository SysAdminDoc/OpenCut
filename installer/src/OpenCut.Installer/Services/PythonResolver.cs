using System.Diagnostics;

namespace OpenCut.Installer.Services;

/// <summary>
/// Finds a supported system Python and returns the interpreter executable,
/// never a PATH-dependent command name or the Python launcher itself.
/// </summary>
public static class PythonResolver
{
    private static readonly string[] PythonCommands = ["python.exe", "python3.exe", "py.exe"];
    private const string VersionProbe =
        "import sys; print(sys.executable); raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)";

    public static string? FindSupportedPath()
    {
        if (!OperatingSystem.IsWindows())
            return null;

        foreach (var command in PythonCommands)
        {
            foreach (var candidate in FindOnPath(command))
            {
                var resolved = ValidatePython(candidate, useLauncher: command.Equals("py.exe", StringComparison.OrdinalIgnoreCase));
                if (resolved is not null)
                    return resolved;
            }
        }

        return null;
    }

    private static IReadOnlyList<string> FindOnPath(string command)
    {
        var candidates = new List<string>();
        Process? process = null;
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = WindowsProcessPaths.Where,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add(command);
            process = Process.Start(startInfo);
            if (process is null)
                return candidates;

            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(10_000))
            {
                try { process.Kill(entireProcessTree: true); }
                catch { /* Best effort; this candidate is unusable. */ }
                return candidates;
            }

            Task.WaitAll(outputTask, errorTask);
            if (process.ExitCode != 0)
                return candidates;

            foreach (var line in outputTask.Result.Split(
                ['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (Path.IsPathFullyQualified(line) && File.Exists(line))
                    candidates.Add(Path.GetFullPath(line));
            }
            return candidates;
        }
        catch
        {
            return candidates;
        }
        finally
        {
            process?.Dispose();
        }
    }

    private static string? ValidatePython(string candidate, bool useLauncher)
    {
        if (!Path.IsPathFullyQualified(candidate) || !File.Exists(candidate))
            return null;

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = candidate,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (useLauncher)
                startInfo.ArgumentList.Add("-3");
            startInfo.ArgumentList.Add("-c");
            startInfo.ArgumentList.Add(VersionProbe);

            using var process = Process.Start(startInfo);
            if (process is null)
                return null;

            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(10_000))
            {
                try { process.Kill(entireProcessTree: true); }
                catch { /* Best effort; this candidate is unusable. */ }
                return null;
            }

            Task.WaitAll(outputTask, errorTask);
            if (process.ExitCode != 0)
                return null;

            var resolved = outputTask.Result
                .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .LastOrDefault();
            if (resolved is null || !Path.IsPathFullyQualified(resolved) || !File.Exists(resolved))
                return null;

            return Path.GetFullPath(resolved);
        }
        catch
        {
            return null;
        }
    }
}
