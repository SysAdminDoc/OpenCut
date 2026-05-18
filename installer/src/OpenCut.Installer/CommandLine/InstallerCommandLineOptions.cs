using OpenCut.Installer.Models;

namespace OpenCut.Installer.CommandLine;

public enum InstallerCommandMode
{
    InteractiveInstall,
    InteractiveUninstall,
    QuietInstall,
    QuietUninstall
}

public sealed class InstallerCommandLineOptions
{
    public InstallerCommandMode Mode { get; private init; } = InstallerCommandMode.InteractiveInstall;
    public InstallConfig Config { get; private init; } = new();
    public string? LogPath { get; private init; }

    public bool IsQuiet =>
        Mode is InstallerCommandMode.QuietInstall or InstallerCommandMode.QuietUninstall;

    public static InstallerCommandLineOptions Parse(IEnumerable<string> rawArgs)
    {
        var args = rawArgs.Where(arg => !string.IsNullOrWhiteSpace(arg)).ToList();
        var config = new InstallConfig();
        string? logPath = null;
        bool install = false;
        bool uninstall = false;
        bool quiet = false;

        for (int i = 0; i < args.Count; i++)
        {
            var (option, inlineValue) = SplitOption(args[i]);
            var normalized = NormalizeOption(option);

            switch (normalized)
            {
                case "--install":
                    install = true;
                    break;
                case "--uninstall":
                    uninstall = true;
                    break;
                case "--quiet":
                case "--silent":
                    quiet = true;
                    break;
                case "--dir":
                case "--install-path":
                    config.InstallPath = RequireValue(args, ref i, option, inlineValue);
                    break;
                case "--log":
                    logPath = RequireValue(args, ref i, option, inlineValue);
                    break;
                case "--user-data-dir":
                    config.UserDataPath = RequireValue(args, ref i, option, inlineValue);
                    break;
                case "--no-desktop-shortcut":
                    config.CreateDesktopShortcut = false;
                    break;
                case "--no-start-menu-shortcut":
                    config.CreateStartMenuShortcut = false;
                    break;
                case "--no-startup-shortcut":
                    config.CreateStartupShortcut = false;
                    break;
                case "--startup-shortcut":
                    config.CreateStartupShortcut = true;
                    break;
                case "--no-cep-extension":
                    config.InstallCepExtension = false;
                    break;
                case "--no-player-debug-mode":
                    config.SetPlayerDebugMode = false;
                    break;
                case "--no-path-update":
                    config.UpdatePath = false;
                    break;
                case "--no-register-uninstaller":
                    config.RegisterUninstaller = false;
                    break;
                case "--download-whisper-model":
                    config.DownloadWhisperModel = true;
                    break;
                case "--no-whisper-model":
                    config.DownloadWhisperModel = false;
                    break;
                case "--whisper-model":
                    config.WhisperModel = RequireValue(args, ref i, option, inlineValue);
                    break;
                case "--optional-deps":
                    config.InstallOptionalDeps = true;
                    config.SelectedDeps = SplitList(RequireValue(args, ref i, option, inlineValue));
                    break;
                case "--no-optional-deps":
                    config.InstallOptionalDeps = false;
                    config.SelectedDeps.Clear();
                    break;
                default:
                    throw new ArgumentException($"Unknown installer argument: {option}");
            }
        }

        if (install && uninstall)
            throw new ArgumentException("Choose either --install or --uninstall, not both.");

        var mode = quiet
            ? uninstall ? InstallerCommandMode.QuietUninstall : InstallerCommandMode.QuietInstall
            : uninstall ? InstallerCommandMode.InteractiveUninstall : InstallerCommandMode.InteractiveInstall;

        return new InstallerCommandLineOptions
        {
            Mode = mode,
            Config = config,
            LogPath = logPath
        };
    }

    private static (string Option, string? Value) SplitOption(string arg)
    {
        var separator = arg.IndexOf('=');
        if (separator <= 0)
            return (arg, null);

        return (arg[..separator], arg[(separator + 1)..]);
    }

    private static string NormalizeOption(string option)
    {
        var normalized = option.Trim().ToLowerInvariant();
        if (normalized.StartsWith('/'))
            normalized = "--" + normalized[1..];
        return normalized;
    }

    private static string RequireValue(List<string> args, ref int index, string option, string? inlineValue)
    {
        if (!string.IsNullOrWhiteSpace(inlineValue))
            return inlineValue;

        if (index + 1 >= args.Count)
            throw new ArgumentException($"{option} requires a value.");

        var value = args[++index];
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"{option} requires a non-empty value.");

        return value;
    }

    private static List<string> SplitList(string value) =>
        value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(item => item.Length > 0)
            .ToList();
}
