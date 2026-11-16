using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class FfmpegSecurityVerifierTests
{
    [Fact]
    public void Release812IsRejected()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner("ffmpeg version 8.1.2-test");

        Assert.False(grade.IsSafe);
        Assert.Equal("release", grade.Lane);
        Assert.Contains("release 8.1.2 is refused", grade.Reason);
    }

    [Theory]
    [InlineData("ffprobe version 8.1.3-essentials_build-www.gyan.dev", "8.1.3")]
    [InlineData("ffmpeg version 9.0.2-full_build-www.gyan.dev", "9.0.2")]
    [InlineData("ffmpeg version n9.1", "9.1.0")]
    public void PublishedReleasesMissTheLateFixesAndAreRejected(string banner, string version)
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(banner);

        Assert.False(grade.IsSafe);
        Assert.Equal("release", grade.Lane);
        Assert.Contains($"release {version} is refused", grade.Reason);
        Assert.Contains("CVE-2026-125064", grade.Reason);
        Assert.Contains("CVE-2026-125180", grade.Reason);
        Assert.Contains("CVE-2026-107675", grade.Reason);
    }

    [Fact]
    public void DatedPostFixSnapshotIsAccepted()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffmpeg version 2026-11-11-git-23c4540053-full_build-www.gyan.dev");

        Assert.True(grade.IsSafe);
        Assert.Equal("snapshot", grade.Lane);
    }

    [Fact]
    public void SnapshotBeforeTheExifFixIsRejected()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffmpeg version 2026-08-03-git-01a25f74cc-full_build-www.gyan.dev");

        Assert.False(grade.IsSafe);
        Assert.Equal("snapshot", grade.Lane);
        Assert.Contains("CVE-2026-125180", grade.Reason);
    }

    [Fact]
    public void PreFloorSnapshotIsRejected()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffmpeg version 2026-05-01-git-aaaaaaaaaa-full_build-www.gyan.dev");

        Assert.False(grade.IsSafe);
        Assert.Equal("snapshot", grade.Lane);
    }
}
