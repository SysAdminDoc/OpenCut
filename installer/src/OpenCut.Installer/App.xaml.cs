using System.Windows;
using System.Windows.Threading;

namespace OpenCut.Installer;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // Global exception handlers — write crash log before dying
        DispatcherUnhandledException += OnDispatcherUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;
        TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;

        base.OnStartup(e);

        // Check for --uninstall flag
        var args = Environment.GetCommandLineArgs();
        bool uninstallMode = args.Length > 1 &&
            args[1].Equals("--uninstall", StringComparison.OrdinalIgnoreCase);

        var mainWindow = new MainWindow(uninstallMode);
        mainWindow.Show();
    }

    private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        WriteCrashLog("DispatcherUnhandledException", e.Exception);
        e.Handled = false;
    }

    private static void OnUnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        WriteCrashLog("UnhandledException", e.ExceptionObject as Exception);
    }

    private static void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        WriteCrashLog("UnobservedTaskException", e.Exception);
    }

    private static void WriteCrashLog(string source, Exception? ex)
    {
        try
        {
            var logPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                "OpenCut-Setup-crash.log");
            var msg = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {source}\n{ex}\n\n";
            File.AppendAllText(logPath, msg);
        }
        catch { /* last resort */ }
    }
}
