namespace OpenCut.Installer.Models;

public static class AppConstants
{
    public const string AppName = "OpenCut";
    public const string AppVersion = "1.12.0";
    public const string AppDisplayName = $"{AppName} v{AppVersion}";
    public const string AppPublisher = "SysAdminDoc";
    public const string AppUrl = "https://github.com/SysAdminDoc/OpenCut";
    public const string AppGuid = "{8A7B9C0D-1E2F-3A4B-5C6D-7E8F9A0B1C2D}";

    public const string DefaultInstallPath = @"C:\Program Files\OpenCut";
    public const string ServerExeName = "OpenCut-Server.exe";
    public const string LauncherVbs = "OpenCut-Launcher.vbs";
    public const int ServerPort = 5679;

    public const string CepExtensionId = "com.opencut.panel";
    public const string PayloadMagic = "OCPAYLOAD";
    public const int PayloadMagicLength = 9;
    public const int PayloadSizeLength = 8;

    // Registry paths
    public const string UninstallRegKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\" + AppGuid;
    public const string AppRegKey = @"Software\OpenCut";
    public const string EnvironmentRegKey = "Environment";

    // CSXS versions for PlayerDebugMode
    public static readonly int[] CsxsVersions = { 7, 8, 9, 10, 11, 12 };

    // Whisper model sizes for display
    public static readonly Dictionary<string, string> WhisperModels = new()
    {
        ["tiny"] = "tiny (75 MB) — Fastest, lower accuracy",
        ["base"] = "base (150 MB) — Good balance",
        ["small"] = "small (500 MB) — Better accuracy, slower",
        ["medium"] = "medium (1.5 GB) — High accuracy, more RAM",
        ["turbo"] = "turbo (1.6 GB) — Best speed/accuracy (recommended)"
    };
}
