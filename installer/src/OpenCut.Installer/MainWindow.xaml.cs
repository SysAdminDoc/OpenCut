using System.Windows;
using System.Windows.Data;
using OpenCut.Installer.Models;
using OpenCut.Installer.Pages;

namespace OpenCut.Installer;

public partial class MainWindow : Window
{
    public InstallConfig Config { get; }
    private readonly bool _uninstallMode;

    public MainWindow(bool uninstallMode = false, InstallConfig? config = null)
    {
        InitializeComponent();
        Config = config ?? new InstallConfig();
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
        // WPF Frame measures Page content at its desired size, which can place
        // a page's fixed action row below a non-resizable shell. Bind every
        // page to the actual viewport so its internal star rows and scrollers
        // receive a finite layout constraint.
        page.SetBinding(
            FrameworkElement.WidthProperty,
            new Binding(nameof(PageFrame.ActualWidth)) { Source = PageFrame });
        page.SetBinding(
            FrameworkElement.HeightProperty,
            new Binding(nameof(PageFrame.ActualHeight)) { Source = PageFrame });
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
