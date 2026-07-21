namespace OpenCut.Installer.Models;

public static class AppConstants
{
    public const string AppName = "OpenCut";
    public const string AppVersion = "1.38.0";
    public const string AppDisplayName = $"{AppName} v{AppVersion}";
    public const string AppPublisher = "SysAdminDoc";
    public const string AppUrl = "https://github.com/SysAdminDoc/OpenCut";
    public const string AppGuid = "{8A7B9C0D-1E2F-3A4B-5C6D-7E8F9A0B1C2D}";
    public const string BundledFfmpegVersion = "8.1.2-essentials_build-www.gyan.dev";
    public const string BundledFfprobeVersion = "8.1.2-essentials_build-www.gyan.dev";
    public const string BundledFfmpegSecurityFloor = "release>=8.1.2 OR git-master>=2026-06-10 (commit b29bdd3715)";
    public const string BundledFfmpegSecurityCve = "CVE-2026-8461";
    public static readonly string[] BundledFfmpegSecurityFixCommits =
    [
        "374b726ffa878ee1cadb987bd1e1e20cc7ed8845",
        "5806e8b9f34f1b0663b3017ef9dd1aa5d08116d1"
    ];
    public const string InstallerManifestFile = "installer.json";

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
