using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;
using Microsoft.Win32;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Pages;

public partial class WelcomePage : Page
{
    private readonly MainWindow _mainWindow;

    public WelcomePage(MainWindow mainWindow)
    {
        InitializeComponent();
        _mainWindow = mainWindow;
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        VersionText.Text = $"v{AppConstants.AppVersion}";
        InstallModeBadgeText.Text = "Fresh install";
        InstallStatusText.Text = "No earlier install was detected. OpenCut will install to the default location and keep the first-run flow simple.";
        InstallPathText.Text = _mainWindow.Config.InstallPath;

        // Animate logo fade-in
        var fadeIn = new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(600))
        {
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut }
        };
        Logo.BeginAnimation(OpacityProperty, fadeIn);

        // Check if already installed
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(AppConstants.AppRegKey);
            if (key?.GetValue("InstallPath") is string path && Directory.Exists(path))
            {
                UninstallLink.Visibility = Visibility.Visible;
                UninstallLink.Content = "Open Uninstaller";
                _mainWindow.Config.InstallPath = path;
                InstallModeBadgeText.Text = "Update ready";
                InstallStatusText.Text = "An existing OpenCut install was found. Setup will refresh the current location and keep your uninstall path intact.";
                InstallPathText.Text = path;
                InstallActionButton.Content = "Update OpenCut";
            }
        }
        catch { /* Not installed */ }
    }

    private void Next_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.SetStep(1);
        _mainWindow.NavigateToPage(new LicensePage(_mainWindow));
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.Close();
    }

    private void UninstallLink_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.StepIndicator.Visibility = Visibility.Collapsed;
        _mainWindow.NavigateToPage(new UninstallPage(_mainWindow));
    }
}
