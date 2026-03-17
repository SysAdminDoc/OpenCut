using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;
using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Pages;

public partial class ProgressPage : Page
{
    private readonly MainWindow _mainWindow;

    public ProgressPage(MainWindow mainWindow)
    {
        InitializeComponent();
        _mainWindow = mainWindow;
        Loaded += OnLoaded;
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        // Start shimmer animation
        StartShimmer();

        var progress = new Progress<InstallProgress>(report =>
        {
            Dispatcher.Invoke(() =>
            {
                StepLabel.Text = report.StepName;
                ProgressBar.Value = report.OverallPercent;
                LogPanel.AppendLog(report.Message, report.Level);
            });
        });

        var engine = new InstallEngine(_mainWindow.Config);

        try
        {
            await Task.Run(() => engine.RunInstall(progress));

            // Success - navigate to complete page
            _mainWindow.SetStep(4);
            _mainWindow.NavigateToPage(new CompletePage(_mainWindow));
        }
        catch (Exception ex)
        {
            LogPanel.AppendLog($"Installation failed: {ex.Message}", LogLevel.Error);
            StepLabel.Text = "Installation failed";
        }
    }

    private void StartShimmer()
    {
        var animation = new DoubleAnimation
        {
            From = -100,
            To = 560,
            Duration = TimeSpan.FromSeconds(1.5),
            RepeatBehavior = RepeatBehavior.Forever,
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseInOut }
        };
        Shimmer.RenderTransform.BeginAnimation(
            System.Windows.Media.TranslateTransform.XProperty, animation);
    }
}
