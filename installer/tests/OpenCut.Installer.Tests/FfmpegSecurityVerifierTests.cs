using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class FfmpegSecurityVerifierTests
{
    [Fact]
    public void Release811IsRejectedForCve20268461()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner("ffmpeg version 8.1.1-test");

        Assert.False(grade.IsSafe);
        Assert.Equal("release", grade.Lane);
        Assert.Contains("predates 8.1.2", grade.Reason);
    }

    [Fact]
    public void Release812IsAccepted()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffprobe version 8.1.2-essentials_build-www.gyan.dev");

        Assert.True(grade.IsSafe);
        Assert.Equal("release", grade.Lane);
    }

    [Fact]
    public void DatedPostFixSnapshotIsAccepted()
    {
        var grade = FfmpegSecurityVerifier.GradeBanner(
            "ffmpeg version 2026-06-10-git-b29bdd3715-full_build-www.gyan.dev");

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
