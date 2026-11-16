namespace OpenCut.Installer.Models;

public static class AppConstants
{
    public const string AppName = "OpenCut";
    public const string AppVersion = "1.56.0";
    public const string AppDisplayName = $"{AppName} v{AppVersion}";
    public const string AppPublisher = "SysAdminDoc";
    public const string AppUrl = "https://github.com/SysAdminDoc/OpenCut";
    public const string AppGuid = "{8A7B9C0D-1E2F-3A4B-5C6D-7E8F9A0B1C2D}";
    public const string BundledFfmpegVersion = "2026-11-11-git-23c4540053-full_build-www.gyan.dev";
    public const string BundledFfprobeVersion = "2026-11-11-git-23c4540053-full_build-www.gyan.dev";
    public const string BundledFfmpegSecurityFloor = "release>=8.1.3 OR git-master>=2026-11-06 (commit 23c454005319dee4a5bcae5c370f75fdeb325230)";
    public const string BundledFfmpegSecurityCve = "CVE-2026-8461, CVE-2026-64832, CVE-2026-64833, CVE-2026-64835, CVE-2026-66041, CVE-2026-125064, CVE-2026-125180, CVE-2026-107675";
    public static readonly string[] BundledFfmpegSecurityFixCommits =
    [
        "374b726ffa878ee1cadb987bd1e1e20cc7ed8845",
        "5806e8b9f34f1b0663b3017ef9dd1aa5d08116d1",
        "4c6217477fc64305055b37d9d1d0d76d30e37f97",
        "6f80e2765492700622596af720534cef33dd31b4",
        "1836ef96846937a6cc2443698a693104f5c0b21e",
        "4da9812e25894fb51d62a8875cfa8eb39b5e20f5",
        "ec887b13a88afbaf44fb28c3045bda5d434da8dc",
        "30fff47764e59f02ac7868bb832ed51a88f949fd",
        "2b822b7fb6ed195546bc9fd9bdb794e98222bdce"
    ];
    public const string BundledFfmpegSourceCommit = "23c454005319dee4a5bcae5c370f75fdeb325230";
    public const string BundledFfmpegSourceSha256 = "81c206f525b8206a5e3406fd0aacd92fd7070dff6a5cf28842b82e65640aa1e1";
    public const string BundledFfmpegPackageUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-2026-11-11-git-23c4540053-full_build.7z";
    public const string BundledFfmpegPackageSha256 = "6fa482b30da6b3a5b51ae4776d1873488a949724d0302aa5e96cad143e483776";
    public const string InstallerManifestFile = "installer.json";

    public const string DefaultInstallPath = @"C:\Program Files\OpenCut";
    public const string ServerExeName = "OpenCut-Server.exe";
    public const string LauncherVbs = "OpenCut-Launcher.vbs";
    public const int ServerPort = 5679;

    public const string CepExtensionId = "com.opencut.panel";

    // Adobe ends ExtendScript support in Premiere Pro in September 2026, and the
    // CEP panel drives every host mutation through it, so the UXP panel has to
    // ship too. Premiere requires the sideload folder to be "<id>_<version>".
    public const string UxpExtensionId = "com.opencut.uxp";
    public const string PayloadMagic = "OCPAYLOAD";
    public const int PayloadMagicLength = 9;
    public const int PayloadSizeLength = 8;

    // Registry paths
    public const string UninstallRegKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\" + AppGuid;
    public const string AppRegKey = @"Software\OpenCut";
    public const string EnvironmentRegKey = "Environment";

    // CSXS versions for PlayerDebugMode.
    // Covers CSXS 7 (CC 2014) through 18 (Premiere 2025+). Stopping at 12 left
    // every Premiere newer than CC 2022 without PlayerDebugMode, so the unsigned
    // CEP panel never loaded. Must stay in lockstep with Install.ps1 and
    // OpenCut.iss — tests/test_installer_csxs_lockstep.py enforces it.
    public static readonly int[] CsxsVersions = { 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };

    public static string CsxsVersionRange =>
        $"{CsxsVersions[0]}-{CsxsVersions[^1]}";

    // Whisper model sizes for display
    public static readonly Dictionary<string, string> WhisperModels = new()
    {
        ["tiny"] = "tiny (75 MB): fastest, lower accuracy",
        ["base"] = "base (150 MB): good balance",
        ["small"] = "small (500 MB): better accuracy, slower",
        ["medium"] = "medium (1.5 GB): high accuracy, more RAM",
        ["turbo"] = "turbo (1.6 GB): best speed/accuracy (recommended)"
    };
}
