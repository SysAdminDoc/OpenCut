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
        StartShimmer();
        LogPanel.AppendLog("Setup initialized. Preparing installation workflow.", LogLevel.Info);

        var progress = new Progress<InstallProgress>(report =>
        {
            Dispatcher.Invoke(() =>
            {
                ProgressStateText.Text = report.StepName;
                StepLabel.Text = report.Message;
                ProgressBar.Value = report.OverallPercent;
                ProgressPercentText.Text = $"{report.OverallPercent:0}%";
                LogPanel.AppendLog(report.Message, report.Level);
            });
        });

        var engine = new InstallEngine(_mainWindow.Config);

        try
        {
            await Task.Run(() => engine.RunInstall(progress));

            ProgressStateText.Text = "Install complete";
            StepLabel.Text = "OpenCut finished installing successfully. Preparing the final summary…";
            ProgressPercentText.Text = "100%";
            _mainWindow.SetStep(4);
            _mainWindow.NavigateToPage(new CompletePage(_mainWindow));
        }
        catch (Exception ex)
        {
            LogPanel.AppendLog($"Installation failed: {ex.Message}", LogLevel.Error);
            ProgressStateText.Text = "Install failed";
            StepLabel.Text = "Setup ran into an error. Review the log, then close the installer and try again.";
            ProgressSummaryText.Text = "OpenCut could not finish setup. The log below shows the last successful action and the error that stopped installation.";
            ProgressPercentText.Text = $"{ProgressBar.Value:0}%";
            Shimmer.Visibility = Visibility.Collapsed;
            CloseBtn.Visibility = Visibility.Visible;
        }
    }

    private void CloseBtn_Click(object sender, RoutedEventArgs e)
    {
        Application.Current.Shutdown();
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
