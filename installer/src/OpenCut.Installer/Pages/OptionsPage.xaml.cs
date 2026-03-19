using System.IO;
using System.Windows;
using System.Windows.Controls;

namespace OpenCut.Installer.Pages;

public partial class OptionsPage : Page
{
    private readonly MainWindow _mainWindow;

    public OptionsPage(MainWindow mainWindow)
    {
        InitializeComponent();
        _mainWindow = mainWindow;
        PathBox.Text = _mainWindow.Config.InstallPath;
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        UpdateDiskSpace();
    }

    private void Browse_Click(object sender, RoutedEventArgs e)
    {
        // Use OpenFolderDialog (WPF/.NET 8+)
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Select install location",
            InitialDirectory = PathBox.Text
        };

        if (dialog.ShowDialog() == true)
        {
            PathBox.Text = dialog.FolderName;
            UpdateDiskSpace();
        }
    }

    private void UpdateDiskSpace()
    {
        try
        {
            var root = Path.GetPathRoot(PathBox.Text);
            if (root != null)
            {
                var drive = new DriveInfo(root);
                var freeGB = drive.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                DiskSpace.Text = $"{freeGB:F1} GB available on {drive.Name}";
            }
        }
        catch
        {
            DiskSpace.Text = "";
        }
    }

    private void WhisperCheck_Changed(object sender, RoutedEventArgs e)
    {
        ModelPanel.Visibility = WhisperCheck.IsChecked == true
            ? Visibility.Visible : Visibility.Collapsed;
    }

    private void OptionalDepsCheck_Changed(object sender, RoutedEventArgs e)
    {
        OptionalDepsPanel.Visibility = OptionalDepsCheck.IsChecked == true
            ? Visibility.Visible : Visibility.Collapsed;
    }

    private void Back_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.SetStep(1);
        _mainWindow.NavigateToPage(new LicensePage(_mainWindow));
    }

    private void Install_Click(object sender, RoutedEventArgs e)
    {
        // Collect config
        var config = _mainWindow.Config;
        config.InstallPath = PathBox.Text.Trim();
        config.InstallCepExtension = CepCheck.IsChecked == true;
        config.SetPlayerDebugMode = DebugModeCheck.IsChecked == true;
        config.CreateDesktopShortcut = DesktopCheck.IsChecked == true;
        config.CreateStartMenuShortcut = StartMenuCheck.IsChecked == true;
        config.CreateStartupShortcut = AutostartCheck.IsChecked == true;
        config.DownloadWhisperModel = WhisperCheck.IsChecked == true;
        config.InstallOptionalDeps = OptionalDepsCheck.IsChecked == true;

        if (config.InstallOptionalDeps)
        {
            config.SelectedDeps.Clear();
            if (DepAutoEditor.IsChecked == true) config.SelectedDeps.Add("auto-editor");
            if (DepEdgeTts.IsChecked == true) config.SelectedDeps.Add("edge-tts");
            if (DepMediapipe.IsChecked == true) config.SelectedDeps.Add("mediapipe");
        }

        if (config.DownloadWhisperModel)
        {
            if (ModelTiny.IsChecked == true) config.WhisperModel = "tiny";
            else if (ModelBase.IsChecked == true) config.WhisperModel = "base";
            else if (ModelSmall.IsChecked == true) config.WhisperModel = "small";
            else if (ModelMedium.IsChecked == true) config.WhisperModel = "medium";
            else config.WhisperModel = "turbo";
        }

        // Validate path
        if (string.IsNullOrWhiteSpace(config.InstallPath))
        {
            MessageBox.Show("Please select an install location.", "OpenCut Setup",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        // Check for invalid path characters
        try
        {
            var fullPath = Path.GetFullPath(config.InstallPath);
            if (fullPath.Length > 200)
            {
                MessageBox.Show("Install path is too long (max 200 characters).", "OpenCut Setup",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
        }
        catch (Exception)
        {
            MessageBox.Show("Install path contains invalid characters.", "OpenCut Setup",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        // Check disk space (need at least 500 MB)
        try
        {
            var root = Path.GetPathRoot(config.InstallPath);
            if (root != null)
            {
                var driveInfo = new DriveInfo(root);
                if (driveInfo.IsReady && driveInfo.AvailableFreeSpace < 500L * 1024 * 1024)
                {
                    MessageBox.Show($"Insufficient disk space on {root}. At least 500 MB required.",
                        "OpenCut Setup", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }
            }
        }
        catch (Exception) { /* DriveInfo may fail on network paths — allow anyway */ }

        _mainWindow.SetStep(3);
        _mainWindow.NavigateToPage(new ProgressPage(_mainWindow));
    }
}
