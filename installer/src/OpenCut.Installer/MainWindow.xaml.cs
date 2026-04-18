using System.Windows;
using OpenCut.Installer.Models;
using OpenCut.Installer.Pages;

namespace OpenCut.Installer;

public partial class MainWindow : Window
{
    public InstallConfig Config { get; } = new();
    private readonly bool _uninstallMode;

    public MainWindow(bool uninstallMode = false)
    {
        InitializeComponent();
        _uninstallMode = uninstallMode;
        WindowVersionText.Text = $"v{AppConstants.AppVersion}";

        if (_uninstallMode)
        {
            Title = "OpenCut Uninstall";
            WindowTitleText.Text = "OpenCut Uninstall";
            NavigateToPage(new UninstallPage(this));
            StepIndicator.Visibility = Visibility.Collapsed;
        }
        else
        {
            Title = "OpenCut Setup";
            WindowTitleText.Text = "OpenCut Setup";
            NavigateToPage(new WelcomePage(this));
            StepIndicator.CurrentStep = 0;
        }
    }

    public void NavigateToPage(System.Windows.Controls.Page page)
    {
        PageFrame.Navigate(page);
    }

    public void SetStep(int step)
    {
        StepIndicator.CurrentStep = step;
    }

    private void MinBtn_Click(object sender, RoutedEventArgs e)
    {
        WindowState = WindowState.Minimized;
    }

    private void CloseBtn_Click(object sender, RoutedEventArgs e)
    {
        Close();
    }
}
