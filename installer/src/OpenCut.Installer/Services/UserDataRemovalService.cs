using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public sealed record UserDataRemovalResult(
    bool SourceExisted,
    string? BackupPath,
    int FileCount);

public sealed class UserDataRemovalService
{
    private const string ManifestDirectory = "__opencut_uninstall__";

    public UserDataRemovalResult? Apply(InstallConfig config)
    {
        if (!config.RemoveUserData)
            return null;

        var sourcePath = NormalizeDirectory(config.UserDataPath);
        EnsureSafeSource(sourcePath, config.InstallPath);
        if (!Directory.Exists(sourcePath))
            return new UserDataRemovalResult(false, null, 0);

        var backupDirectory = NormalizeDirectory(config.UserDataBackupDirectory);
        if (IsSameOrChild(backupDirectory, sourcePath))
            throw new InvalidOperationException(
                "The user-data backup directory must be outside the directory being removed.");

        Directory.CreateDirectory(backupDirectory);
        var files = EnumerateSafeFiles(sourcePath);
        var stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
        var token = Guid.NewGuid().ToString("N")[..8];
        var backupPath = Path.Combine(backupDirectory, $"OpenCut-user-data-{stamp}-{token}.zip");
        var partialPath = backupPath + ".partial";

        try
        {
            var manifestEntry = CreateBackup(sourcePath, files, partialPath);
            ValidateBackup(partialPath, manifestEntry);
            File.Move(partialPath, backupPath);
            EnsureFilesUnlocked(files);
            RemoveSourceAtomically(sourcePath, backupPath);
            return new UserDataRemovalResult(true, backupPath, files.Count);
        }
        catch
        {
            TryDelete(partialPath);
            throw;
        }
    }

    private static string CreateBackup(
        string sourcePath,
        IReadOnlyList<string> files,
        string partialPath)
    {
        var manifestEntry = $"{ManifestDirectory}/{Guid.NewGuid():N}.json";
        var records = new List<BackupFileRecord>(files.Count);

        using (var stream = new FileStream(
                   partialPath, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None))
        using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: false))
        {
            foreach (var file in files)
            {
                var relativePath = Path.GetRelativePath(sourcePath, file).Replace('\\', '/');
                if (relativePath.StartsWith("../", StringComparison.Ordinal) ||
                    relativePath.Equals("..", StringComparison.Ordinal))
                {
                    throw new InvalidOperationException("A user-data file resolved outside the backup root.");
                }

                var entry = archive.CreateEntry(relativePath, CompressionLevel.Optimal);
                using var input = new FileStream(
                    file, FileMode.Open, FileAccess.Read, FileShare.Read,
                    bufferSize: 128 * 1024, FileOptions.SequentialScan);
                using var output = entry.Open();
                using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                var buffer = new byte[128 * 1024];
                long size = 0;
                int read;
                while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                {
                    output.Write(buffer, 0, read);
                    hash.AppendData(buffer, 0, read);
                    size += read;
                }

                records.Add(new BackupFileRecord(
                    relativePath,
                    size,
                    Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant()));
            }

            var manifest = new BackupManifest(
                SchemaVersion: 1,
                CreatedUtc: DateTime.UtcNow,
                SourcePath: sourcePath,
                Files: records);
            var metadata = archive.CreateEntry(manifestEntry, CompressionLevel.Optimal);
            using var metadataStream = metadata.Open();
            JsonSerializer.Serialize(metadataStream, manifest);
        }

        return manifestEntry;
    }

    private static void ValidateBackup(string partialPath, string manifestEntry)
    {
        if (!File.Exists(partialPath) || new FileInfo(partialPath).Length == 0)
            throw new InvalidDataException("The user-data backup was not created.");

        using var archive = ZipFile.OpenRead(partialPath);
        var manifestArchiveEntry = archive.GetEntry(manifestEntry)
            ?? throw new InvalidDataException("The user-data backup manifest is missing.");
        using var manifestStream = manifestArchiveEntry.Open();
        var manifest = JsonSerializer.Deserialize<BackupManifest>(manifestStream)
            ?? throw new InvalidDataException("The user-data backup manifest is invalid.");

        if (manifest.SchemaVersion != 1)
            throw new InvalidDataException("The user-data backup schema is unsupported.");

        foreach (var record in manifest.Files)
        {
            var entry = archive.GetEntry(record.Path)
                ?? throw new InvalidDataException($"Backup entry is missing: {record.Path}");
            using var entryStream = entry.Open();
            var hash = SHA256.HashData(entryStream);
            if (entry.Length != record.Size ||
                !Convert.ToHexString(hash).Equals(record.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"Backup entry failed validation: {record.Path}");
            }
        }
    }

    private static List<string> EnumerateSafeFiles(string sourcePath)
    {
        var files = new List<string>();
        var pending = new Stack<DirectoryInfo>();
        pending.Push(new DirectoryInfo(sourcePath));

        while (pending.Count > 0)
        {
            var directory = pending.Pop();
            foreach (var entry in directory.EnumerateFileSystemInfos()
                         .OrderBy(item => item.FullName, StringComparer.OrdinalIgnoreCase))
            {
                if ((entry.Attributes & FileAttributes.ReparsePoint) != 0)
                {
                    throw new InvalidOperationException(
                        $"User-data backup refuses reparse points: {entry.FullName}");
                }

                if (entry is DirectoryInfo childDirectory)
                    pending.Push(childDirectory);
                else if (entry is FileInfo file)
                    files.Add(file.FullName);
            }
        }

        files.Sort(StringComparer.OrdinalIgnoreCase);
        return files;
    }

    private static void EnsureFilesUnlocked(IEnumerable<string> files)
    {
        foreach (var file in files)
        {
            using var stream = new FileStream(
                file, FileMode.Open, FileAccess.Read, FileShare.None,
                bufferSize: 1, FileOptions.None);
        }
    }

    private static void RemoveSourceAtomically(string sourcePath, string backupPath)
    {
        var parent = Directory.GetParent(sourcePath)?.FullName
            ?? throw new InvalidOperationException("The user-data root cannot be removed.");
        var leaf = Path.GetFileName(sourcePath);
        var removalPath = Path.Combine(parent, $".{leaf}.opencut-removing-{Guid.NewGuid():N}");

        Directory.Move(sourcePath, removalPath);
        try
        {
            Directory.Delete(removalPath, recursive: true);
        }
        catch (Exception deleteError)
        {
            try
            {
                if (!Directory.Exists(sourcePath) && Directory.Exists(removalPath))
                    Directory.Move(removalPath, sourcePath);
            }
            catch (Exception restoreError)
            {
                throw new IOException(
                    $"User-data removal failed after backup {backupPath}. Remaining data is at " +
                    $"{removalPath}; automatic restore also failed.",
                    new AggregateException(deleteError, restoreError));
            }

            throw new IOException(
                $"User-data removal failed after backup {backupPath}; the data directory was restored.",
                deleteError);
        }
    }

    private static void EnsureSafeSource(string sourcePath, string installPath)
    {
        var root = Path.GetPathRoot(sourcePath);
        if (string.IsNullOrWhiteSpace(root) || PathsEqual(sourcePath, root))
            throw new InvalidOperationException("A filesystem root cannot be used as the OpenCut data path.");

        var profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrWhiteSpace(profile) && PathsEqual(sourcePath, NormalizeDirectory(profile)))
            throw new InvalidOperationException("The user profile root cannot be used as the OpenCut data path.");

        if (!string.IsNullOrWhiteSpace(installPath))
        {
            var normalizedInstallPath = NormalizeDirectory(installPath);
            if (IsSameOrChild(normalizedInstallPath, sourcePath) ||
                IsSameOrChild(sourcePath, normalizedInstallPath))
            {
                throw new InvalidOperationException(
                    "The OpenCut data path must be separate from the application install path.");
            }
        }
    }

    private static string NormalizeDirectory(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("A non-empty directory path is required.", nameof(path));

        var fullPath = Path.GetFullPath(path);
        var root = Path.GetPathRoot(fullPath);
        return root is not null && PathsEqual(fullPath, root)
            ? root
            : Path.TrimEndingDirectorySeparator(fullPath);
    }

    private static bool IsSameOrChild(string candidate, string root)
    {
        if (PathsEqual(candidate, root))
            return true;

        var relative = Path.GetRelativePath(root, candidate);
        return relative != ".." &&
               !relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) &&
               !Path.IsPathRooted(relative);
    }

    private static bool PathsEqual(string left, string right) =>
        string.Equals(
            Path.TrimEndingDirectorySeparator(left),
            Path.TrimEndingDirectorySeparator(right),
            OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal);

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch
        {
            // Best effort cleanup; the final archive is never promoted on failure.
        }
    }

    private sealed record BackupFileRecord(string Path, long Size, string Sha256);

    private sealed record BackupManifest(
        int SchemaVersion,
        DateTime CreatedUtc,
        string SourcePath,
        List<BackupFileRecord> Files);
}
