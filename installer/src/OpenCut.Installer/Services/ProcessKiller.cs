using System.Diagnostics;
using System.Net;
using System.Net.NetworkInformation;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class ProcessKiller
{
    public void KillOpenCutProcesses(IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Report(progress, step, totalSteps, "Checking for running OpenCut processes...");

        // Kill OpenCut-Server.exe by name
        KillByName("OpenCut-Server", progress, step, totalSteps);

        // Kill any wscript instances running our launcher
        KillWscriptLaunchers(progress, step, totalSteps);

        // Kill anything on port 5679
        KillByPort(AppConstants.ServerPort, progress, step, totalSteps);

        // Brief pause to let processes die
        Thread.Sleep(500);

        Report(progress, step, totalSteps, "Process cleanup complete.", LogLevel.Success);
    }

    private void KillByName(string processName, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        try
        {
            var procs = Process.GetProcessesByName(processName);
            foreach (var proc in procs)
            {
                try
                {
                    Report(progress, step, totalSteps, $"Killing {proc.ProcessName} (PID {proc.Id})...");
                    proc.Kill(entireProcessTree: true);
                    proc.WaitForExit(3000);
                    Report(progress, step, totalSteps, $"Killed {proc.ProcessName}.", LogLevel.Success);
                }
                catch (Exception ex)
                {
                    Report(progress, step, totalSteps, $"Could not kill {proc.ProcessName}: {ex.Message}", LogLevel.Warning);
                }
                finally
                {
                    proc.Dispose();
                }
            }
        }
        catch { /* Process enumeration failed */ }
    }

    private void KillWscriptLaunchers(IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        try
        {
            foreach (var proc in Process.GetProcessesByName("wscript"))
            {
                try
                {
                    var cmdLine = GetCommandLine(proc);
                    if (cmdLine != null && cmdLine.Contains("OpenCut", StringComparison.OrdinalIgnoreCase))
                    {
                        Report(progress, step, totalSteps, $"Killing wscript launcher (PID {proc.Id})...");
                        proc.Kill();
                        proc.WaitForExit(2000);
                    }
                }
                catch { /* Best effort */ }
                finally { proc.Dispose(); }
            }
        }
        catch { /* Process enumeration failed */ }
    }

    private void KillByPort(int port, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        try
        {
            var props = IPGlobalProperties.GetIPGlobalProperties();
            var listeners = props.GetActiveTcpListeners();
            var listening = listeners.Any(ep => ep.Port == port);

            if (!listening) return;

            // Use netstat to find PID
            var psi = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = $"/c netstat -ano | findstr :{port} | findstr LISTENING",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var netstat = Process.Start(psi);
            if (netstat == null) return;

            var output = netstat.StandardOutput.ReadToEnd();
            netstat.WaitForExit(5000);

            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 5 && int.TryParse(parts[^1], out var pid) && pid > 0)
                {
                    try
                    {
                        using var proc = Process.GetProcessById(pid);
                        Report(progress, step, totalSteps, $"Killing process on port {port} (PID {pid})...");
                        proc.Kill(entireProcessTree: true);
                        proc.WaitForExit(3000);
                    }
                    catch { /* Process may have already exited */ }
                }
            }
        }
        catch { /* Best effort */ }
    }

    private static string? GetCommandLine(Process process)
    {
        try
        {
            // Use PowerShell CIM instead of deprecated wmic (removed in Win11 23H2+)
            var psi = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = $"-NoProfile -Command \"(Get-CimInstance Win32_Process -Filter 'ProcessId={process.Id}').CommandLine\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using var p = Process.Start(psi);
            if (p == null) return null;
            var output = p.StandardOutput.ReadToEnd();
            p.WaitForExit(5000);
            return output;
        }
        catch { return null; }
    }

    private static void Report(IProgress<InstallProgress> progress, int step, int total,
        string message, LogLevel level = LogLevel.Info)
    {
        progress.Report(new InstallProgress
        {
            StepNumber = step,
            TotalSteps = total,
            StepName = "Stopping OpenCut processes",
            Message = message,
            Level = level
        });
    }
}
