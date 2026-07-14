namespace OpenCut.Installer.Models;

public class InstallConfig
{
    public string InstallPath { get; set; } = AppConstants.DefaultInstallPath;
    public bool CreateDesktopShortcut { get; set; } = true;
    public bool CreateStartMenuShortcut { get; set; } = true;
    public bool CreateStartupShortcut { get; set; }
    public bool InstallCepExtension { get; set; } = true;
    public bool SetPlayerDebugMode { get; set; } = true;
    public bool UpdatePath { get; set; } = true;
    public bool RegisterUninstaller { get; set; } = true;
    public bool DownloadWhisperModel { get; set; }
    public string WhisperModel { get; set; } = "turbo";
    public bool InstallOptionalDeps { get; set; }
    public List<string> SelectedDeps { get; set; } = [];
    public string UserDataPath { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".opencut");
    public bool RemoveUserData { get; set; }
    public string UserDataBackupDirectory { get; set; } = GetDefaultUserDataBackupDirectory();

    public string ServerPath => Path.Combine(InstallPath, "server");
    public string FfmpegPath => Path.Combine(InstallPath, "ffmpeg");
    public string ExtensionPath => Path.Combine(InstallPath, "extension", AppConstants.CepExtensionId);
    public string LogsPath => Path.Combine(InstallPath, "logs");
    public string UninstallExePath => Path.Combine(InstallPath, "OpenCut-Uninstall.exe");
    public string InstallerManifestPath => Path.Combine(UserDataPath, AppConstants.InstallerManifestFile);

    public string CepTargetPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "Adobe", "CEP", "extensions", AppConstants.CepExtensionId);

    private static string GetDefaultUserDataBackupDirectory()
    {
        var documents = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        var parent = string.IsNullOrWhiteSpace(documents)
            ? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
            : documents;
        return Path.Combine(parent, "OpenCut Backups");
    }
}
