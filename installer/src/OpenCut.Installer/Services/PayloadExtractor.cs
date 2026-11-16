using System.IO.Compression;
using System.Text;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public class PayloadExtractor
{
    /// <summary>
    /// Extract the payload ZIP to the given directory.
    /// Tries self-extracting mode first (appended data), then adjacent payload.zip.
    /// </summary>
    public void Extract(string targetDir, IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        Directory.CreateDirectory(targetDir);

        // Try self-extracting mode
        var exePath = Environment.ProcessPath;
        if (exePath != null && TryExtractAppended(exePath, targetDir, progress, step, totalSteps))
            return;

        // Fallback: adjacent payload.zip
        var adjacentZip = FindAdjacentPayload();
        if (adjacentZip != null)
        {
            Report(progress, step, totalSteps, $"Extracting {Path.GetFileName(adjacentZip)}...", LogLevel.Info);
            ZipFile.ExtractToDirectory(adjacentZip, targetDir, overwriteFiles: true);
            Report(progress, step, totalSteps, "Payload extracted.", LogLevel.Success);
            return;
        }

        throw new FileNotFoundException(
            "No payload found. Expected appended data or adjacent payload.zip.");
    }

    private bool TryExtractAppended(string exePath, string targetDir,
        IProgress<InstallProgress> progress, int step, int totalSteps)
    {
        try
        {
            using var fs = new FileStream(exePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (!TryFindAppended(fs, out var payloadOffset, out var payloadSize))
                return false;

            Report(progress, step, totalSteps, "Extracting embedded payload...", LogLevel.Info);

            using var subStream = new SubStream(fs, payloadOffset, payloadSize);
            using var archive = new ZipArchive(subStream, ZipArchiveMode.Read);

            var fullTargetDir = Path.GetFullPath(targetDir);

            foreach (var entry in archive.Entries)
            {
                var destPath = Path.GetFullPath(Path.Combine(targetDir, entry.FullName));

                // ZIP Slip prevention: ensure extracted path stays within target directory
                if (!destPath.StartsWith(fullTargetDir + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                    && !destPath.Equals(fullTargetDir, StringComparison.OrdinalIgnoreCase))
                {
                    continue; // Skip malicious entry
                }

                if (entry.FullName.EndsWith('/') || entry.FullName.EndsWith('\\'))
                {
                    Directory.CreateDirectory(destPath);
                    continue;
                }

                var dir = Path.GetDirectoryName(destPath);
                if (dir != null) Directory.CreateDirectory(dir);

                entry.ExtractToFile(destPath, overwrite: true);
            }

            Report(progress, step, totalSteps, "Embedded payload extracted.", LogLevel.Success);
            return true;
        }
        catch
        {
            return false;
        }
    }

    // Trailer: [payload bytes][8-byte LE size][9-byte "OCPAYLOAD" magic]
    private static bool TryFindAppended(FileStream fs, out long payloadOffset, out long payloadSize)
    {
        payloadOffset = payloadSize = 0;
        var trailerSize = AppConstants.PayloadMagicLength + AppConstants.PayloadSizeLength;
        if (fs.Length < trailerSize)
            return false;

        fs.Seek(-AppConstants.PayloadMagicLength, SeekOrigin.End);
        var magicBuf = new byte[AppConstants.PayloadMagicLength];
        fs.ReadExactly(magicBuf);
        if (Encoding.ASCII.GetString(magicBuf) != AppConstants.PayloadMagic)
            return false;

        fs.Seek(-trailerSize, SeekOrigin.End);
        var sizeBuf = new byte[AppConstants.PayloadSizeLength];
        fs.ReadExactly(sizeBuf);
        payloadSize = BitConverter.ToInt64(sizeBuf);
        payloadOffset = fs.Length - trailerSize - payloadSize;
        return payloadSize > 0 && payloadOffset >= 0;
    }

    /// <summary>
    /// The text of one small file at the payload's root, read from the
    /// payload appended to <paramref name="exePath"/> or the payload.zip next
    /// to it, without extracting anything. Null when there's no payload, no
    /// such entry, or it's larger than <paramref name="maxBytes"/>.
    /// </summary>
    internal static string? ReadEntryText(string? exePath, string entryName, int maxBytes = 4096)
    {
        if (exePath == null) return null;
        try
        {
            using (var fs = new FileStream(exePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (TryFindAppended(fs, out var offset, out var size))
                {
                    using var subStream = new SubStream(fs, offset, size);
                    using var archive = new ZipArchive(subStream, ZipArchiveMode.Read);
                    return ReadEntry(archive, entryName, maxBytes);
                }
            }
            var adjacent = AdjacentPayload(exePath);
            if (adjacent == null) return null;
            using var zip = ZipFile.OpenRead(adjacent);
            return ReadEntry(zip, entryName, maxBytes);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static string? ReadEntry(ZipArchive archive, string entryName, int maxBytes)
    {
        var entry = archive.GetEntry(entryName);
        if (entry == null || entry.Length > maxBytes) return null;
        using var reader = new StreamReader(entry.Open(), Encoding.UTF8);
        return reader.ReadToEnd();
    }

    private static string? AdjacentPayload(string exePath)
    {
        var dir = Path.GetDirectoryName(exePath);
        if (dir == null) return null;

        var zipPath = Path.Combine(dir, "payload.zip");
        return File.Exists(zipPath) ? zipPath : null;
    }

    private string? FindAdjacentPayload()
    {
        var exePath = Environment.ProcessPath;
        return exePath == null ? null : AdjacentPayload(exePath);
    }

    private static void Report(IProgress<InstallProgress> progress, int step, int total,
        string message, LogLevel level)
    {
        progress.Report(new InstallProgress
        {
            StepNumber = step,
            TotalSteps = total,
            StepName = "Extracting payload",
            Message = message,
            Level = level
        });
    }
}
