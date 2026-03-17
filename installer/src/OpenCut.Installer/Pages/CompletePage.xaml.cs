using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Pages;

public partial class CompletePage : Page
{
    private readonly MainWindow _mainWindow;

    public CompletePage(MainWindow mainWindow)
    {
        InitializeComponent();
        _mainWindow = mainWindow;
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        var config = _mainWindow.Config;

        PathSummary.Text = $"Installed to: {config.InstallPath}";

        var components = new List<string>();
        if (config.InstallCepExtension) components.Add("CEP Extension");
        if (config.CreateDesktopShortcut) components.Add("Desktop shortcut");
        if (config.CreateStartMenuShortcut) components.Add("Start Menu");
        if (config.CreateStartupShortcut) components.Add("Autostart");
        if (config.DownloadWhisperModel) components.Add($"Whisper ({config.WhisperModel})");
        ComponentsSummary.Text = components.Count > 0
            ? string.Join(" | ", components) : "Server + FFmpeg";

        // Animate checkmark
        var fadeIn = new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(400))
        {
            BeginTime = TimeSpan.FromMilliseconds(200),
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut }
        };
        CheckPath.BeginAnimation(OpacityProperty, fadeIn);
    }

    private void Close_Click(object sender, RoutedEventArgs e)
    {
        if (LaunchCheck.IsChecked == true)
        {
            try
            {
                var vbsPath = Path.Combine(_mainWindow.Config.InstallPath, AppConstants.LauncherVbs);
                if (File.Exists(vbsPath))
                {
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = "wscript.exe",
                        Arguments = $"\"{vbsPath}\"",
                        UseShellExecute = true
                    });
                }
                else
                {
                    var serverExe = Path.Combine(_mainWindow.Config.ServerPath, AppConstants.ServerExeName);
                    if (File.Exists(serverExe))
                    {
                        Process.Start(new ProcessStartInfo
                        {
                            FileName = serverExe,
                            WorkingDirectory = _mainWindow.Config.ServerPath,
                            UseShellExecute = true
                        });
                    }
                }
            }
            catch { /* Best effort */ }
        }

        _mainWindow.Close();
    }
}
