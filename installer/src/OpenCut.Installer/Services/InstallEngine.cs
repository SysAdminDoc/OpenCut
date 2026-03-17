using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

/// <summary>
/// Orchestrates the full install sequence, replicating all Inno Setup operations.
/// </summary>
public class InstallEngine
{
    private readonly InstallConfig _config;
    private readonly ProcessKiller _processKiller = new();
    private readonly PayloadExtractor _payloadExtractor = new();
    private readonly FileInstaller _fileInstaller = new();
    private readonly RegistryManager _registryManager = new();
    private readonly ShortcutCreator _shortcutCreator = new();
    private readonly CepInstaller _cepInstaller = new();
    private readonly WhisperDownloader _whisperDownloader = new();
    private readonly DependencyInstaller _dependencyInstaller = new();

    public InstallEngine(InstallConfig config)
    {
        _config = config;
    }

    public void RunInstall(IProgress<InstallProgress> progress)
    {
        int totalSteps = 17;
        int step = 0;

        var tempDir = Path.Combine(Path.GetTempPath(), $"OpenCut-Install-{Guid.NewGuid():N}");

        try
        {
            // Step 1: Kill existing processes
            step = 1;
            _processKiller.KillOpenCutProcesses(progress, step, totalSteps);

            // Step 2: Extract payload
            step = 2;
            _payloadExtractor.Extract(tempDir, progress, step, totalSteps);

            // Step 3: Copy server files
            step = 3;
            Report(progress, step, totalSteps, "Copying server files", "Installing server...");
            var serverSrc = Path.Combine(tempDir, "server");
            if (Directory.Exists(serverSrc))
                _fileInstaller.CopyDirectory(serverSrc, _config.ServerPath,
                    "Copying server files", progress, step, totalSteps);
            else
            {
                // Try OpenCut-Server directory name (PyInstaller default)
                var altSrc = Path.Combine(tempDir, "OpenCut-Server");
                if (Directory.Exists(altSrc))
                    _fileInstaller.CopyDirectory(altSrc, _config.ServerPath,
                        "Copying server files", progress, step, totalSteps);
                else
                    Report(progress, step, totalSteps, "Copying server files",
                        "Server files not found in payload.", LogLevel.Warning);
            }

            // Step 4: Copy FFmpeg
            step = 4;
            Report(progress, step, totalSteps, "Copying FFmpeg", "Installing FFmpeg...");
            var ffmpegSrc = Path.Combine(tempDir, "ffmpeg");
            if (Directory.Exists(ffmpegSrc))
                _fileInstaller.CopyDirectory(ffmpegSrc, _config.FfmpegPath,
                    "Copying FFmpeg", progress, step, totalSteps);

            // Step 5: Copy CEP extension to install dir
            step = 5;
            Report(progress, step, totalSteps, "Copying extension", "Copying CEP extension...");
            var extSrc = Path.Combine(tempDir, "extension", AppConstants.CepExtensionId);
            if (!Directory.Exists(extSrc))
                extSrc = Path.Combine(tempDir, AppConstants.CepExtensionId);
            if (Directory.Exists(extSrc))
                _fileInstaller.CopyDirectory(extSrc, _config.ExtensionPath,
                    "Copying extension", progress, step, totalSteps);

            // Step 6: Copy launcher VBS + logo.ico
            step = 6;
            Report(progress, step, totalSteps, "Copying launcher", "Installing launcher...");
            var vbsSrc = Path.Combine(tempDir, AppConstants.LauncherVbs);
            if (File.Exists(vbsSrc))
                _fileInstaller.CopyFile(vbsSrc,
                    Path.Combine(_config.InstallPath, AppConstants.LauncherVbs),
                    "Copying launcher", progress, step, totalSteps);

            var icoSrc = Path.Combine(tempDir, "logo.ico");
            if (File.Exists(icoSrc))
                _fileInstaller.CopyFile(icoSrc,
                    Path.Combine(_config.InstallPath, "logo.ico"),
                    "Copying icon", progress, step, totalSteps);

            // Step 7: Create logs directory
            step = 7;
            _fileInstaller.EnsureDirectory(_config.LogsPath,
                "Creating directories", progress, step, totalSteps);

            // Step 8: Add FFmpeg to PATH
            step = 8;
            _registryManager.AddToPath(_config.FfmpegPath, progress, step, totalSteps);

            // Step 9: Set PlayerDebugMode
            step = 9;
            if (_config.SetPlayerDebugMode)
                _registryManager.SetPlayerDebugMode(progress, step, totalSteps);
            else
                Report(progress, step, totalSteps, "PlayerDebugMode", "Skipped (not selected).", LogLevel.Debug);

            // Step 10: Install CEP extension to Adobe folder
            step = 10;
            if (_config.InstallCepExtension && Directory.Exists(_config.ExtensionPath))
                _cepInstaller.InstallExtension(_config, progress, step, totalSteps);
            else
                Report(progress, step, totalSteps, "CEP extension", "Skipped (not selected).", LogLevel.Debug);

            // Step 11: Write install path to registry
            step = 11;
            _registryManager.WriteInstallPath(_config.InstallPath, progress, step, totalSteps);

            // Step 12: Create shortcuts
            step = 12;
            _shortcutCreator.CreateShortcuts(_config, progress, step, totalSteps);

            // Step 13: Copy installer as uninstaller
            step = 13;
            Report(progress, step, totalSteps, "Copying uninstaller", "Installing uninstaller...");
            var selfPath = Environment.ProcessPath;
            if (selfPath != null && File.Exists(selfPath))
            {
                Directory.CreateDirectory(_config.InstallPath);
                File.Copy(selfPath, _config.UninstallExePath, overwrite: true);
                Report(progress, step, totalSteps, "Copying uninstaller",
                    "Uninstaller installed.", LogLevel.Success);
            }

            // Step 14: Register in Add/Remove Programs
            step = 14;
            _registryManager.RegisterUninstall(_config, progress, step, totalSteps);

            // Step 15: Download Whisper model (optional)
            step = 15;
            if (_config.DownloadWhisperModel)
                _whisperDownloader.DownloadModel(_config, progress, step, totalSteps);
            else
                Report(progress, step, totalSteps, "Whisper model", "Skipped (not selected).", LogLevel.Debug);

            // Step 16: Install optional Python tools
            step = 16;
            _dependencyInstaller.InstallDeps(_config, progress, step, totalSteps);

            // Step 17: Cleanup temp
            step = 17;
            Report(progress, step, totalSteps, "Cleaning up", "Removing temporary files...");
            try
            {
                if (Directory.Exists(tempDir))
                    Directory.Delete(tempDir, recursive: true);
                Report(progress, step, totalSteps, "Cleaning up", "Temporary files removed.", LogLevel.Success);
            }
            catch (Exception ex)
            {
                Report(progress, step, totalSteps, "Cleaning up",
                    $"Could not remove temp dir: {ex.Message}", LogLevel.Warning);
            }

            Report(progress, totalSteps, totalSteps, "Installation complete",
                "OpenCut has been installed successfully!", LogLevel.Success);
        }
        catch (Exception ex)
        {
            // Cleanup temp on failure
            try { if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true); }
            catch { /* Ignore cleanup errors */ }

            throw new InvalidOperationException($"Installation failed at step {step}: {ex.Message}", ex);
        }
    }

    private static void Report(IProgress<InstallProgress> progress, int step, int total,
        string stepName, string message, LogLevel level = LogLevel.Info)
    {
        progress.Report(new InstallProgress
        {
            StepNumber = step,
            TotalSteps = total,
            StepName = stepName,
            Message = message,
            Level = level
        });
    }
}
