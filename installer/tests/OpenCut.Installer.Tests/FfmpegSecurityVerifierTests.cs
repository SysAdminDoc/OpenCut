using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class FfmpegSecurityVerifierTests
{
    [Fact]
    public void Release812IsRejectedForJuly2026Cves()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner("ffmpeg version 8.1.2-test");

        Assert.False(grade.IsSafe);
        Assert.Equal("release", grade.Lane);
        Assert.Contains("predates 8.1.3", grade.Reason);
        Assert.Contains("CVE-2026-64832", grade.Reason);
    }

    [Fact]
    public void Release813ReenablesReleaseLane()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffprobe version 8.1.3-essentials_build-www.gyan.dev");

        Assert.True(grade.IsSafe);
        Assert.Equal("release", grade.Lane);
    }

    [Fact]
    public void DatedPostFixSnapshotIsAccepted()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffmpeg version 2026-08-03-git-01a25f74cc-full_build-www.gyan.dev");

        Assert.True(grade.IsSafe);
        Assert.Equal("snapshot", grade.Lane);
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
