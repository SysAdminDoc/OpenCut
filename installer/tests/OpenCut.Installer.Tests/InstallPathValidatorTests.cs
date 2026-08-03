using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class InstallPathValidatorTests
{
    [Fact]
    public void EmptyPathIsRejected()
    {
        var result = InstallPathValidator.Validate("   ");

        Assert.Equal(InstallPathVerdict.Rejected, result.Verdict);
        Assert.Contains("install location", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RelativePathIsRejectedRatherThanResolvedAgainstTheWorkingDirectory()
    {
        var result = InstallPathValidator.Validate(@"OpenCut\bin");

        Assert.Equal(InstallPathVerdict.Rejected, result.Verdict);
    }

    [Fact]
    public void DriveRootGetsItsOwnFolderAndIsNeverStoredBare()
    {
        var result = InstallPathValidator.Validate(@"D:\");

        Assert.NotEqual(InstallPathVerdict.Rejected, result.Verdict);
        Assert.Equal(@"D:\OpenCut", result.NormalizedPath);
        Assert.NotEmpty(result.Message);
    }

    [Fact]
    public void OverlongPathIsRejected()
    {
        var longPath = @"C:\" + new string('a', InstallPathValidator.MaxPathLength + 10);

        Assert.Equal(InstallPathVerdict.Rejected, InstallPathValidator.Validate(longPath).Verdict);
    }

    [Fact]
    public void NormalizedPathIsAlwaysAbsoluteAndSeparatorTrimmed()
    {
        // A path that does not exist on the test machine, so the verdict turns
        // only on normalisation and not on whatever is already on disk.
        var target = @"C:\Program Files\OpenCutPathNormalizationTest";
        var result = InstallPathValidator.Validate(target + @"\");

        Assert.Equal(InstallPathVerdict.Accepted, result.Verdict);
        Assert.Equal(target, result.NormalizedPath);
        Assert.True(Path.IsPathFullyQualified(result.NormalizedPath));
    }

    [Fact]
    public void SystemAndProfileDirectoriesAreRejectedAsInstallRoots()
    {
        foreach (var folder in new[]
                 {
                     Environment.SpecialFolder.Windows,
                     Environment.SpecialFolder.System,
                     Environment.SpecialFolder.ProgramFiles,
                     Environment.SpecialFolder.UserProfile,
                     Environment.SpecialFolder.MyVideos,
                 })
        {
            var path = Environment.GetFolderPath(folder);
            if (string.IsNullOrWhiteSpace(path)) continue;

            Assert.Equal(InstallPathVerdict.Rejected, InstallPathValidator.Validate(path).Verdict);
        }
    }

    [Fact]
    public void ANonEmptyForeignDirectoryNeedsConfirmation()
    {
        var temp = Path.Combine(Path.GetTempPath(), "OpenCutPathTest_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            File.WriteAllText(Path.Combine(temp, "wedding-master.mov"), "not really a video");

            var result = InstallPathValidator.Validate(temp);

            Assert.Equal(InstallPathVerdict.NeedsConfirmation, result.Verdict);
            Assert.Equal(temp, result.NormalizedPath);
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public void AnEmptyOrPreviousInstallDirectoryIsAcceptedWithoutPrompting()
    {
        var temp = Path.Combine(Path.GetTempPath(), "OpenCutPathTest_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        try
        {
            Assert.Equal(InstallPathVerdict.Accepted, InstallPathValidator.Validate(temp).Verdict);

            Directory.CreateDirectory(Path.Combine(temp, "server"));
            Directory.CreateDirectory(Path.Combine(temp, "ffmpeg"));

            Assert.Equal(InstallPathVerdict.Accepted, InstallPathValidator.Validate(temp).Verdict);
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public void UninstallRefusesDriveRootsSystemDirectoriesAndTheUserProfile()
    {
        Assert.False(InstallPathValidator.IsSafeToDelete(null));
        Assert.False(InstallPathValidator.IsSafeToDelete(""));
        Assert.False(InstallPathValidator.IsSafeToDelete(@"D:\"));
        Assert.False(InstallPathValidator.IsSafeToDelete(@"C:\"));
        Assert.False(InstallPathValidator.IsSafeToDelete(@"OpenCut"));
        Assert.False(InstallPathValidator.IsSafeToDelete(
            Environment.GetFolderPath(Environment.SpecialFolder.Windows)));
        Assert.False(InstallPathValidator.IsSafeToDelete(
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "System32")));
        Assert.False(InstallPathValidator.IsSafeToDelete(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)));
        Assert.False(InstallPathValidator.IsSafeToDelete(
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles)));
    }

    [Fact]
    public void UninstallStillDeletesADedicatedInstallFolder()
    {
        Assert.True(InstallPathValidator.IsSafeToDelete(@"C:\Program Files\OpenCut"));
        Assert.True(InstallPathValidator.IsSafeToDelete(@"D:\OpenCut\"));
        Assert.True(InstallPathValidator.IsSafeToDelete(
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "OpenCut")));
    }
}
