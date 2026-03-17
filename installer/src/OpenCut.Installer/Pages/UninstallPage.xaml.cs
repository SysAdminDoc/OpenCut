using System.Windows;
using System.Windows.Controls;
using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Pages;

public partial class UninstallPage : Page
{
    private readonly MainWindow _mainWindow;

    public UninstallPage(MainWindow mainWindow)
    {
        InitializeComponent();
        _mainWindow = mainWindow;
    }

    private async void Uninstall_Click(object sender, RoutedEventArgs e)
    {
        UninstallBtn.IsEnabled = false;
        CancelBtn.IsEnabled = false;
        ProgressBar.Visibility = Visibility.Visible;
        StatusText.Text = "Removing OpenCut...";

        var progress = new Progress<InstallProgress>(report =>
        {
            Dispatcher.Invoke(() =>
            {
                StatusText.Text = report.StepName;
                ProgressBar.Value = report.OverallPercent;
                LogPanel.AppendLog(report.Message, report.Level);
            });
        });

        var engine = new UninstallEngine(_mainWindow.Config);

        try
        {
            await Task.Run(() => engine.RunUninstall(progress));

            HeaderText.Text = "Uninstall Complete";
            StatusText.Text = "OpenCut has been removed from your system.";
            UninstallBtn.Visibility = Visibility.Collapsed;
            CancelBtn.Content = "Close";
            CancelBtn.IsEnabled = true;

            LogPanel.AppendLog("Uninstall completed successfully.", LogLevel.Success);

            // Schedule self-delete if running as uninstaller
            engine.ScheduleSelfDelete();
        }
        catch (Exception ex)
        {
            LogPanel.AppendLog($"Uninstall failed: {ex.Message}", LogLevel.Error);
            StatusText.Text = "Uninstall encountered errors.";
            CancelBtn.IsEnabled = true;
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.Close();
    }
}
