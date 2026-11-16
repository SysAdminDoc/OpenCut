using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using OpenCut.Installer.Models;

namespace OpenCut.Installer.Services;

public sealed record FfmpegSecurityGrade(
    bool IsSafe,
    string Lane,
    string Version,
    string Reason);

public sealed partial class FfmpegSecurityVerifier
{
    private static readonly DateOnly SnapshotFloor = new(2026, 11, 6);

    // No published FFmpeg release clears the advisory matrix. 8.1.3 and 9.0.2
    // were checked fix by fix and both miss these three fixes, and a higher
    // version number is not evidence, so every tagged release is refused. Keep
    // in step with RELEASE_LANE_CLOSED_REASON in opencut/core/ffmpeg_provenance.py
    // (tests/test_ffmpeg_release_lane.py checks the CVEs and commits).
    internal const string ReleaseLaneClosedReason =
        "no published FFmpeg release carries the fixes for CVE-2026-125064 (ec887b13a8), " +
        "CVE-2026-125180 (30fff47764) and CVE-2026-107675 (2b822b7fb6); " +
        "8.1.3 and 9.0.2 both lack them";

    public void VerifyPayload(string ffmpegDirectory)
    {
        if (!Directory.Exists(ffmpegDirectory))
            throw new InvalidDataException("The installer payload has no FFmpeg directory.");

        VerifyBinary(Path.Combine(ffmpegDirectory, "ffmpeg.exe"));
        VerifyBinary(Path.Combine(ffmpegDirectory, "ffprobe.exe"));
    }

    public FfmpegSecurityGrade VerifyBinary(string binaryPath)
    {
        if (!File.Exists(binaryPath))
            throw new FileNotFoundException("Required media binary is missing.", binaryPath);

        var startInfo = new ProcessStartInfo
        {
            FileName = binaryPath,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        startInfo.ArgumentList.Add("-version");

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException($"Could not start {binaryPath}.");
        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(15_000))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException($"Version verification timed out for {binaryPath}.");
        }

        Task.WaitAll(stdoutTask, stderrTask);
        var banner = stdoutTask.Result.Length > 0 ? stdoutTask.Result : stderrTask.Result;
        var grade = GradeBanner(banner);
        if (process.ExitCode != 0 || !grade.IsSafe)
        {
            throw new InvalidDataException(
                $"Unsafe FFmpeg payload ({Path.GetFileName(binaryPath)}): {grade.Reason}. " +
                $"OpenCut requires {AppConstants.BundledFfmpegSecurityFloor} for " +
                $"{AppConstants.BundledFfmpegSecurityCve}.");
        }

        return grade;
    }

    public static FfmpegSecurityGrade GradeBanner(string banner)
    {
        var firstLine = (banner ?? string.Empty)
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault() ?? string.Empty;
        var tokenMatch = VersionTokenRegex().Match(firstLine);
        if (!tokenMatch.Success)
            return new(false, "unknown", string.Empty, "version banner could not be parsed");

        var token = tokenMatch.Groups[1].Value;
        var snapshotMatch = SnapshotDateRegex().Match(token);
        if (snapshotMatch.Success &&
            DateOnly.TryParseExact(
                snapshotMatch.Groups[1].Value,
                "yyyy-MM-dd",
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out var snapshotDate))
        {
            var safe = snapshotDate >= SnapshotFloor;
            return new(
                safe,
                "snapshot",
                token,
                safe
                    ? $"snapshot {snapshotDate:yyyy-MM-dd} is at/after {SnapshotFloor:yyyy-MM-dd}"
                    : $"snapshot {snapshotDate:yyyy-MM-dd} predates {SnapshotFloor:yyyy-MM-dd}, when the last graded fix (CVE-2026-125180) landed");
        }

        var releaseMatch = ReleaseRegex().Match(token);
        if (!releaseMatch.Success)
            return new(false, "unknown", token, "build is neither a dated snapshot nor a release");

        var release = new Version(
            int.Parse(releaseMatch.Groups[1].Value, CultureInfo.InvariantCulture),
            int.Parse(releaseMatch.Groups[2].Value, CultureInfo.InvariantCulture),
            releaseMatch.Groups[3].Success
                ? int.Parse(releaseMatch.Groups[3].Value, CultureInfo.InvariantCulture)
                : 0);
        return new(false, "release", token, $"release {release} is refused: {ReleaseLaneClosedReason}");
    }

    [GeneratedRegex(@"\bversion\s+([^\s]+)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex VersionTokenRegex();

    [GeneratedRegex(@"^(\d{4}-\d{2}-\d{2})-git-[0-9a-f]{7,40}", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex SnapshotDateRegex();

    [GeneratedRegex(@"^n?(\d+)\.(\d+)(?:\.(\d+))?", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex ReleaseRegex();
}
