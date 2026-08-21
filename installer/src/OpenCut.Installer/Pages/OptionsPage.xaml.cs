using System.IO;
using System.Windows;
using System.Windows.Controls;
using OpenCut.Installer.Services;

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
        UpdateFormState();
        UpdateDiskSpace();
        UpdateSelectionSummary();
    }

    private void Browse_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Select install location",
            InitialDirectory = PathBox.Text
        };

        if (dialog.ShowDialog() == true)
        {
            PathBox.Text = dialog.FolderName;
            UpdateDiskSpace();
            UpdateSelectionSummary();
        }
    }

    private void PathBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (PathValidationPanel != null)
            PathValidationPanel.Visibility = Visibility.Collapsed;
        UpdateDiskSpace();
        UpdateSelectionSummary();
    }

    private void WhisperCheck_Changed(object sender, RoutedEventArgs e)
    {
        UpdateFormState();
        UpdateSelectionSummary();
    }

    private void OptionalDepsCheck_Changed(object sender, RoutedEventArgs e)
    {
        UpdateFormState();
        UpdateSelectionSummary();
    }

    private void CepCheck_Changed(object sender, RoutedEventArgs e)
    {
        UpdateFormState();
        UpdateSelectionSummary();
    }

    private void AnyOptionChanged(object sender, RoutedEventArgs e)
    {
        UpdateSelectionSummary();
    }

    private void UpdateFormState()
    {
        OptionalDepsPanel.Visibility = OptionalDepsCheck.IsChecked == true
            ? Visibility.Visible
            : Visibility.Collapsed;

        ModelPanel.Visibility = WhisperCheck.IsChecked == true
            ? Visibility.Visible
            : Visibility.Collapsed;

        var cepEnabled = CepCheck.IsChecked == true;
        DebugModeCheck.IsEnabled = cepEnabled;
        if (!cepEnabled) DebugModeCheck.IsChecked = false;

        CepStatusText.Text = cepEnabled
            ? (DebugModeCheck.IsChecked == true
                ? "The CEP panel will be installed and Adobe debug mode will be enabled so the extension can load reliably."
                : "The CEP panel will be installed, but Adobe debug mode will stay off until you enable it manually.")
            : "Premiere CEP integration will be skipped. You can still run the OpenCut server and add the panel later.";
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
                DiskSpace.Text = $"{freeGB:F1} GB available on {drive.Name} • Estimated setup footprint {GetEstimatedInstallSizeLabel()}";
                return;
            }
        }
        catch
        {
            // Ignore and fall through.
        }

        DiskSpace.Text = $"Estimated setup footprint {GetEstimatedInstallSizeLabel()}";
    }

    private void UpdateSelectionSummary()
    {
        DestinationSummaryText.Text = PathBox.Text.Trim();

        if (CepCheck.IsChecked == true)
        {
            IntegrationSummaryText.Text = DebugModeCheck.IsChecked == true
                ? "CEP panel + debug mode"
                : "CEP panel only";
        }
        else
        {
            IntegrationSummaryText.Text = "Server only";
        }

        var extras = new List<string>();

        if (WhisperCheck.IsChecked == true)
        {
            extras.Add($"Whisper {GetSelectedWhisperModel()}");
        }

        if (OptionalDepsCheck.IsChecked == true)
        {
            var depCount = 0;
            if (DepAutoEditor.IsChecked == true) depCount++;
            if (DepEdgeTts.IsChecked == true) depCount++;
            if (DepMediapipe.IsChecked == true) depCount++;
            extras.Add(depCount > 0
                ? $"{depCount} optional tool{(depCount == 1 ? "" : "s")}"
                : "Optional tools enabled");
        }

        var shortcutParts = new List<string>();
        if (DesktopCheck.IsChecked == true) shortcutParts.Add("Desktop");
        if (StartMenuCheck.IsChecked == true) shortcutParts.Add("Start Menu");
        if (AutostartCheck.IsChecked == true) shortcutParts.Add("Startup");

        if (shortcutParts.Count > 0)
        {
            extras.Add(string.Join(" + ", shortcutParts));
        }

        ExtrasSummaryText.Text = extras.Count > 0
            ? string.Join(" • ", extras)
            : "Core install only";
    }

    private string GetSelectedWhisperModel()
    {
        if (ModelTiny.IsChecked == true) return "tiny";
        if (ModelBase.IsChecked == true) return "base";
        if (ModelSmall.IsChecked == true) return "small";
        if (ModelMedium.IsChecked == true) return "medium";
        return "turbo";
    }

    private string GetEstimatedInstallSizeLabel()
    {
        var estimateMb = 350;

        if (WhisperCheck.IsChecked == true)
        {
            estimateMb += GetSelectedWhisperModel() switch
            {
                "tiny" => 75,
                "base" => 150,
                "small" => 500,
                "medium" => 1500,
                _ => 1600,
            };
        }

        if (OptionalDepsCheck.IsChecked == true)
        {
            if (DepAutoEditor.IsChecked == true) estimateMb += 45;
            if (DepEdgeTts.IsChecked == true) estimateMb += 20;
            if (DepMediapipe.IsChecked == true) estimateMb += 140;
        }

        return estimateMb >= 1024
            ? $"{estimateMb / 1024.0:F1} GB"
            : $"{estimateMb} MB";
    }

    private void Back_Click(object sender, RoutedEventArgs e)
    {
        _mainWindow.SetStep(1);
        _mainWindow.NavigateToPage(new LicensePage(_mainWindow));
    }

    private void Install_Click(object sender, RoutedEventArgs e)
    {
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
            config.WhisperModel = GetSelectedWhisperModel();
        }

        // Normalise once here. Everything downstream, including the
        // recursive delete the uninstaller performs, trusts InstallPath.
        var pathCheck = InstallPathValidator.Validate(config.InstallPath);
        if (pathCheck.Verdict != InstallPathVerdict.Accepted)
        {
            ShowPathValidation(pathCheck.Message);
            return;
        }

        config.InstallPath = pathCheck.NormalizedPath;
        PathBox.Text = pathCheck.NormalizedPath;

        try
        {
            var root = Path.GetPathRoot(config.InstallPath);
            if (root != null)
            {
                var driveInfo = new DriveInfo(root);
                if (driveInfo.IsReady && driveInfo.AvailableFreeSpace < 500L * 1024 * 1024)
                {
                    ShowPathValidation($"Insufficient disk space on {root}. At least 500 MB is required.");
                    return;
                }
            }
        }
        catch (Exception)
        {
            // DriveInfo may fail on network paths. Allow them to continue.
        }

        _mainWindow.SetStep(3);
        _mainWindow.NavigateToPage(new ProgressPage(_mainWindow));
    }

    private void ShowPathValidation(string message)
    {
        PathValidationText.Text = message;
        PathValidationPanel.Visibility = Visibility.Visible;
        PathBox.Focus();
        PathBox.SelectAll();
    }
}
