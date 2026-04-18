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
        VersionText.Text = $"v{AppConstants.AppVersion}";
        SummaryText.Text = $"{AppConstants.AppName} has been installed successfully. Your local editing workflow is ready for the first launch.";

        PathSummary.Text = $"Installed to: {config.InstallPath}";

        var components = new List<string>();
        if (config.InstallCepExtension) components.Add("CEP Extension");
        if (config.CreateDesktopShortcut) components.Add("Desktop shortcut");
        if (config.CreateStartMenuShortcut) components.Add("Start Menu");
        if (config.CreateStartupShortcut) components.Add("Autostart");
        if (config.DownloadWhisperModel) components.Add($"Whisper ({config.WhisperModel})");
        ComponentsSummary.Text = components.Count > 0
            ? string.Join(" • ", components) : "Server + FFmpeg";

        NextStepsText.Text = config.InstallCepExtension
            ? "1. Launch the OpenCut Server.\n2. Open Premiere Pro and load the OpenCut panel.\n3. Start with a clip, transcript, or cleanup pass."
            : "1. Launch the OpenCut Server.\n2. Open the workflow surface you plan to use first.\n3. Add Premiere integration later if you want the CEP panel available.";
        UpdateFinishButtonLabel();

        // Animate checkmark
        var fadeIn = new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(400))
        {
            BeginTime = TimeSpan.FromMilliseconds(200),
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut }
        };
        CheckPath.BeginAnimation(OpacityProperty, fadeIn);
    }

    private void LaunchCheck_Changed(object sender, RoutedEventArgs e)
    {
        UpdateFinishButtonLabel();
    }

    private void UpdateFinishButtonLabel()
    {
        FinishButton.Content = LaunchCheck.IsChecked == true ? "Launch & Close" : "Finish";
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
