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

            // Read trailer: [payload bytes][8-byte LE size][9-byte "OCPAYLOAD" magic]
            if (fs.Length < AppConstants.PayloadMagicLength + AppConstants.PayloadSizeLength)
                return false;

            // Read magic
            fs.Seek(-AppConstants.PayloadMagicLength, SeekOrigin.End);
            var magicBuf = new byte[AppConstants.PayloadMagicLength];
            fs.ReadExactly(magicBuf);
            var magic = Encoding.ASCII.GetString(magicBuf);
            if (magic != AppConstants.PayloadMagic)
                return false;

            // Read size
            fs.Seek(-(AppConstants.PayloadMagicLength + AppConstants.PayloadSizeLength), SeekOrigin.End);
            var sizeBuf = new byte[AppConstants.PayloadSizeLength];
            fs.ReadExactly(sizeBuf);
            var payloadSize = BitConverter.ToInt64(sizeBuf);

            // Calculate offset
            var trailerSize = AppConstants.PayloadMagicLength + AppConstants.PayloadSizeLength;
            var payloadOffset = fs.Length - trailerSize - payloadSize;
            if (payloadOffset < 0)
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

    private string? FindAdjacentPayload()
    {
        var exePath = Environment.ProcessPath;
        if (exePath == null) return null;

        var dir = Path.GetDirectoryName(exePath);
        if (dir == null) return null;

        var zipPath = Path.Combine(dir, "payload.zip");
        return File.Exists(zipPath) ? zipPath : null;
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
