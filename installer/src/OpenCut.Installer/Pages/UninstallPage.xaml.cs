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
        RemoveUserDataCheckBox.IsChecked = _mainWindow.Config.RemoveUserData;
        UserDataPathText.Text =
            $"Data path: {_mainWindow.Config.UserDataPath}. Verified backup folder: " +
            $"{_mainWindow.Config.UserDataBackupDirectory}.";
    }

    private async void Uninstall_Click(object sender, RoutedEventArgs e)
    {
        var removeUserData = RemoveUserDataCheckBox.IsChecked == true;
        if (removeUserData)
        {
            var confirmation = MessageBox.Show(
                "Remove all OpenCut user data after creating and validating a backup?\n\n" +
                $"Data: {_mainWindow.Config.UserDataPath}\n" +
                $"Backup: {_mainWindow.Config.UserDataBackupDirectory}\n\n" +
                "This includes settings, jobs, journals, indexes, plugins, models, and project/agent state.",
                "Confirm OpenCut data removal",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning,
                MessageBoxResult.No);
            if (confirmation != MessageBoxResult.Yes)
                return;
        }

        _mainWindow.Config.RemoveUserData = removeUserData;
        UninstallBtn.IsEnabled = false;
        CancelBtn.IsEnabled = false;
        RemoveUserDataCheckBox.IsEnabled = false;
        ProgressBar.Visibility = Visibility.Visible;
        StatusText.Text = "Removing OpenCut…";

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
            StatusText.Text = "OpenCut has been removed from this system.";
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
            StatusText.Text = "Uninstall encountered errors. Review the log, then close the installer when you are ready.";
            CancelBtn.IsEnabled = true;
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.Close();
    }
}
