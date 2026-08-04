using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

/// <summary>
/// Captures the state touched by an install so a failure can restore the
/// previous install instead of leaving a partially replaced tree behind.
/// </summary>
public sealed class InstallTransaction : IDisposable
{
    private readonly InstallConfig _config;
    private readonly RegistryManager? _registryManager;
    private readonly bool _captureExternalState;
    private readonly string _rollbackRoot = Path.Combine(
        Path.GetTempPath(), $"OpenCut-Rollback-{Guid.NewGuid():N}");
    private readonly List<DirectoryBackup> _directoryBackups = [];
    private readonly List<FileBackup> _fileBackups = [];
    private RegistryInstallState? _registryState;
    private bool _begun;
    private bool _completed;

    public InstallTransaction(
        InstallConfig config,
        RegistryManager? registryManager = null,
        bool captureExternalState = true)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _registryManager = registryManager;
        _captureExternalState = captureExternalState;
    }

    public string? PreviousInstallPath { get; private set; }

    public void Begin(IProgress<InstallProgress> progress, int totalSteps)
    {
        if (_begun)
            throw new InvalidOperationException("The install transaction has already started.");

        Directory.CreateDirectory(_rollbackRoot);
        _begun = true;

        if (_registryManager is not null)
        {
            _registryState = _registryManager.CaptureInstallState();
            PreviousInstallPath = RegistryManager.GetInstalledPath();
        }

        BackupDirectory(_config.InstallPath, "install tree");
        if (_captureExternalState)
        {
            BackupDirectory(_config.CepTargetPath, "CEP extension");
            foreach (var shortcutPath in ShortcutCreator.GetShortcutPaths())
                BackupFile(shortcutPath, "shortcut");
            BackupFile(_config.InstallerManifestPath, "installer manifest");
        }

        if (!string.IsNullOrWhiteSpace(PreviousInstallPath))
            BackupDirectory(PreviousInstallPath, "previous install tree");

        Report(progress, 1, totalSteps, "Preparing installation",
            PreviousInstallPath is null
                ? "Prepared a rollback snapshot for a new installation."
                : $"Detected the previous installation at {PreviousInstallPath}; rollback snapshot is ready.",
            LogLevel.Debug);
    }

    public void PrepareCleanInstallRoot()
    {
        EnsureBegun();

        DeleteDirectoryIfPresent(_config.InstallPath);
        if (!string.IsNullOrWhiteSpace(PreviousInstallPath) &&
            !PathsEqual(PreviousInstallPath, _config.InstallPath))
        {
            DeleteDirectoryIfPresent(PreviousInstallPath);
        }

        Directory.CreateDirectory(_config.InstallPath);
    }

    public void Commit(IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        EnsureBegun();
        _completed = true;
        TryDeleteRollbackRoot(progress, step, totalSteps, LogLevel.Warning);
    }

    public void Rollback(IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        if (!_begun || _completed)
            return;

        var failures = new List<Exception>();
        Report(progress, step, totalSteps, "Rolling back installation",
            "Restoring the previous installation and integration state.", LogLevel.Warning);

        foreach (var backup in _directoryBackups.AsEnumerable().Reverse())
        {
            try
            {
                RestoreDirectory(backup);
            }
            catch (Exception ex)
            {
                failures.Add(new IOException(
                    $"Could not restore {backup.Label} at {backup.OriginalPath}: {ex.Message}", ex));
            }
        }

        foreach (var backup in _fileBackups.AsEnumerable().Reverse())
        {
            try
            {
                RestoreFile(backup);
            }
            catch (Exception ex)
            {
                failures.Add(new IOException(
                    $"Could not restore {backup.Label} at {backup.OriginalPath}: {ex.Message}", ex));
            }
        }

        if (_registryManager is not null && _registryState is not null)
        {
            try
            {
                _registryManager.RestoreInstallState(_registryState);
            }
            catch (Exception ex)
            {
                failures.Add(new InvalidOperationException(
                    $"Could not restore installer registry state: {ex.Message}", ex));
            }
        }

        if (failures.Count > 0)
        {
            Report(progress, step, totalSteps, "Rollback incomplete",
                $"Rollback could not restore {failures.Count} item(s). Review the installer log before retrying.",
                LogLevel.Error);
            TryDeleteRollbackRoot(progress, step, totalSteps, LogLevel.Warning);
            throw new AggregateException("The installation failed and rollback was incomplete.", failures);
        }

        _completed = true;
        TryDeleteRollbackRoot(progress, step, totalSteps, LogLevel.Warning);
        Report(progress, step, totalSteps, "Rollback complete",
            "The machine was restored to its pre-install state.", LogLevel.Success);
    }

    public void Dispose()
    {
        if (!_completed)
            return;

        TryDeleteRollbackRoot(null, 0, 0, LogLevel.Warning);
    }

    private void BackupDirectory(string path, string label)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        var fullPath = Path.GetFullPath(path)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (_directoryBackups.Any(item => PathsEqual(item.OriginalPath, fullPath)))
            return;
        if (PathsEqual(fullPath, _rollbackRoot))
            throw new InvalidOperationException("The rollback directory cannot be inside the install tree.");

        var backupPath = Path.Combine(_rollbackRoot, $"directory-{_directoryBackups.Count:D2}");
        var existed = Directory.Exists(fullPath);
        if (existed)
            CopyDirectory(fullPath, backupPath);
        else if (File.Exists(fullPath))
            throw new IOException($"The install path is a file, not a directory: {fullPath}");
        _directoryBackups.Add(new DirectoryBackup(fullPath, backupPath, existed, label));
    }

    private void BackupFile(string path, string label)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        var fullPath = Path.GetFullPath(path);
        if (_fileBackups.Any(item => PathsEqual(item.OriginalPath, fullPath)))
            return;

        var existed = File.Exists(fullPath);
        var backupPath = Path.Combine(_rollbackRoot, $"file-{_fileBackups.Count:D2}");
        if (existed)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(backupPath)!);
            File.Copy(fullPath, backupPath, overwrite: true);
        }
        _fileBackups.Add(new FileBackup(fullPath, backupPath, existed, label));
    }

    private static void RestoreDirectory(DirectoryBackup backup)
    {
        DeleteDirectoryIfPresent(backup.OriginalPath);
        if (backup.Existed)
            CopyDirectory(backup.BackupPath, backup.OriginalPath);
    }

    private static void RestoreFile(FileBackup backup)
    {
        if (File.Exists(backup.OriginalPath))
            File.Delete(backup.OriginalPath);
        if (backup.Existed)
        {
            var parent = Path.GetDirectoryName(backup.OriginalPath);
            if (!string.IsNullOrWhiteSpace(parent))
                Directory.CreateDirectory(parent);
            File.Copy(backup.BackupPath, backup.OriginalPath, overwrite: true);
        }
    }

    private static void DeleteDirectoryIfPresent(string path)
    {
        if (Directory.Exists(path))
            Directory.Delete(path, recursive: true);
        else if (File.Exists(path))
            File.Delete(path);
    }

    private static void CopyDirectory(string source, string destination)
    {
        if (!Directory.Exists(source))
            throw new DirectoryNotFoundException(source);

        Directory.CreateDirectory(destination);
        foreach (var directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(source, directory);
            Directory.CreateDirectory(Path.Combine(destination, relative));
        }

        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(source, file);
            var target = Path.Combine(destination, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(file, target, overwrite: true);
        }
    }

    private void TryDeleteRollbackRoot(
        IProgress<InstallProgress>? progress,
        int step,
        int totalSteps,
        LogLevel failureLevel)
    {
        if (!Directory.Exists(_rollbackRoot))
            return;

        try
        {
            Directory.Delete(_rollbackRoot, recursive: true);
        }
        catch (Exception ex)
        {
            if (progress is not null)
            {
                Report(progress, step, totalSteps, "Rollback snapshot cleanup",
                    $"Could not remove rollback snapshot: {ex.Message}", failureLevel);
            }
        }
    }

    private void EnsureBegun()
    {
        if (!_begun)
            throw new InvalidOperationException("The install transaction has not started.");
    }

    private static bool PathsEqual(string left, string right)
    {
        var normalizedLeft = Path.GetFullPath(left)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var normalizedRight = Path.GetFullPath(right)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return string.Equals(normalizedLeft, normalizedRight, StringComparison.OrdinalIgnoreCase);
    }

    private static void Report(
        IProgress<InstallProgress> progress,
        int step,
        int total,
        string stepName,
        string message,
        LogLevel level)
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

    private sealed record DirectoryBackup(
        string OriginalPath,
        string BackupPath,
        bool Existed,
        string Label);

    private sealed record FileBackup(string OriginalPath, string BackupPath, bool Existed, string Label);
}
