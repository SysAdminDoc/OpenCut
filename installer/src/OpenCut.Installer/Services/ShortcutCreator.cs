using System.Runtime.InteropServices;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class ShortcutCreator
{
    public void CreateShortcuts(InstallConfig config, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        var stepName = "Creating shortcuts";
        var iconPath = Path.Combine(config.InstallPath, "logo.ico");
        var vbsPath = Path.Combine(config.InstallPath, AppConstants.LauncherVbs);
        var serverExe = Path.Combine(config.ServerPath, AppConstants.ServerExeName);

        // Desktop shortcut
        if (config.CreateDesktopShortcut)
        {
            var desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
            var linkPath = Path.Combine(desktopPath, "OpenCut.lnk");
            CreateShortcut(linkPath, "wscript.exe", $"\"{vbsPath}\"",
                config.InstallPath, iconPath, "Launch OpenCut Server");
            Report(progress, step, totalSteps, stepName, "Desktop shortcut created.", LogLevel.Success);
        }

        // Start Menu shortcuts
        if (config.CreateStartMenuShortcut)
        {
            var startMenu = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Programs), "OpenCut");
            Directory.CreateDirectory(startMenu);

            // Main shortcut (hidden launcher)
            CreateShortcut(Path.Combine(startMenu, "OpenCut Server.lnk"),
                "wscript.exe", $"\"{vbsPath}\"",
                config.InstallPath, iconPath, "Launch OpenCut Server");

            // Console shortcut
            CreateShortcut(Path.Combine(startMenu, "OpenCut Server (Console).lnk"),
                serverExe, "",
                config.ServerPath, iconPath, "Launch OpenCut Server with console");

            // Uninstall shortcut
            CreateShortcut(Path.Combine(startMenu, "Uninstall OpenCut.lnk"),
                config.UninstallExePath, "--uninstall",
                config.InstallPath, null, "Uninstall OpenCut");

            Report(progress, step, totalSteps, stepName, "Start Menu shortcuts created.", LogLevel.Success);
        }

        // Startup shortcut
        if (config.CreateStartupShortcut)
        {
            var startupPath = Environment.GetFolderPath(Environment.SpecialFolder.Startup);
            var linkPath = Path.Combine(startupPath, "OpenCut.lnk");
            CreateShortcut(linkPath, "wscript.exe", $"\"{vbsPath}\"",
                config.InstallPath, iconPath, "OpenCut Server - Autostart");
            Report(progress, step, totalSteps, stepName, "Startup shortcut created.", LogLevel.Success);
        }
    }

    public void RemoveShortcuts()
    {
        // Desktop
        var desktop = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "OpenCut.lnk");
        TryDelete(desktop);

        // Start Menu folder
        var startMenu = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Programs), "OpenCut");
        if (Directory.Exists(startMenu))
        {
            try { Directory.Delete(startMenu, recursive: true); }
            catch { /* Best effort */ }
        }

        // Startup
        var startup = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Startup), "OpenCut.lnk");
        TryDelete(startup);
    }

    private static void CreateShortcut(string linkPath, string targetPath, string arguments,
        string workingDir, string? iconPath, string description)
    {
        // Use COM WScript.Shell to create .lnk
        var shellType = Type.GetTypeFromProgID("WScript.Shell")
            ?? throw new InvalidOperationException("WScript.Shell COM object not available.");

        dynamic shell = Activator.CreateInstance(shellType)!;
        try
        {
            var shortcut = shell.CreateShortcut(linkPath);
            try
            {
                shortcut.TargetPath = targetPath;
                shortcut.Arguments = arguments;
                shortcut.WorkingDirectory = workingDir;
                shortcut.Description = description;
                if (iconPath != null && File.Exists(iconPath))
                    shortcut.IconLocation = $"{iconPath},0";
                shortcut.Save();
            }
            finally
            {
                Marshal.ReleaseComObject(shortcut);
            }
        }
        finally
        {
            Marshal.ReleaseComObject(shell);
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* Best effort */ }
    }

    private static void Report(IProgress<InstallProgress> progress, int step, int total,
        string stepName, string message, LogLevel level)
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
