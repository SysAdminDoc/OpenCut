using System.IO.Compression;
using System.Text;
using OpenCut.Installer.Models;
using OpenCut.Installer.Services;

namespace OpenCut.Installer.Tests;

public class PayloadEntryTextTests
{
    private const string Requirement = "64-bit (x64) Python 3.13";

    private static byte[] Zip(IDictionary<string, string> entries)
    {
        using var buffer = new MemoryStream();
        using (var archive = new ZipArchive(buffer, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, text) in entries)
            {
                using var writer = new StreamWriter(archive.CreateEntry(name).Open(), new UTF8Encoding(false));
                writer.Write(text);
            }
        }
        return buffer.ToArray();
    }

    // The builder's layout: [exe][payload zip][8-byte LE size]["OCPAYLOAD"].
    private static string AppendedExe(string dir, byte[] payload)
    {
        var exe = Path.Combine(dir, "OpenCut-Setup.exe");
        using var fs = File.Create(exe);
        fs.Write(Encoding.ASCII.GetBytes("MZ fake setup exe"));
        fs.Write(payload);
        fs.Write(BitConverter.GetBytes((long)payload.Length));
        fs.Write(Encoding.ASCII.GetBytes(AppConstants.PayloadMagic));
        return exe;
    }

    [Fact]
    public void ReadsAnEntryFromThePayloadAppendedToTheExe()
    {
        var dir = TestPaths.CreateTempDirectory();
        var exe = AppendedExe(dir, Zip(new Dictionary<string, string>
        {
            ["LICENSE"] = "MIT",
            [DependencyInstaller.PythonRequirementEntry] = Requirement + "\n",
        }));

        Assert.Equal(Requirement + "\n", PayloadExtractor.ReadEntryText(exe, DependencyInstaller.PythonRequirementEntry));
    }

    [Fact]
    public void ReadsAnEntryFromThePayloadZipNextToTheExe()
    {
        var dir = TestPaths.CreateTempDirectory();
        var exe = Path.Combine(dir, "OpenCut-Setup.exe");
        File.WriteAllText(exe, "MZ fake setup exe with nothing appended");
        File.WriteAllBytes(Path.Combine(dir, "payload.zip"), Zip(new Dictionary<string, string>
        {
            [DependencyInstaller.PythonRequirementEntry] = Requirement,
        }));

        Assert.Equal(Requirement, PayloadExtractor.ReadEntryText(exe, DependencyInstaller.PythonRequirementEntry));
    }

    [Fact]
    public void AMissingEntryOrPayloadReadsAsNothing()
    {
        var dir = TestPaths.CreateTempDirectory();
        var appended = AppendedExe(dir, Zip(new Dictionary<string, string> { ["LICENSE"] = "MIT" }));
        var bare = Path.Combine(TestPaths.CreateTempDirectory(), "OpenCut-Setup.exe");
        File.WriteAllText(bare, "MZ no payload anywhere");

        Assert.Null(PayloadExtractor.ReadEntryText(appended, DependencyInstaller.PythonRequirementEntry));
        Assert.Null(PayloadExtractor.ReadEntryText(bare, DependencyInstaller.PythonRequirementEntry));
        Assert.Null(PayloadExtractor.ReadEntryText(Path.Combine(dir, "gone.exe"), DependencyInstaller.PythonRequirementEntry));
        Assert.Null(PayloadExtractor.ReadEntryText(null, DependencyInstaller.PythonRequirementEntry));
    }

    [Fact]
    public void AnEntryLargerThanTheLimitIsNotRead()
    {
        var dir = TestPaths.CreateTempDirectory();
        var exe = AppendedExe(dir, Zip(new Dictionary<string, string>
        {
            [DependencyInstaller.PythonRequirementEntry] = new string('x', 5000),
        }));

        Assert.Null(PayloadExtractor.ReadEntryText(exe, DependencyInstaller.PythonRequirementEntry));
    }

    [Fact]
    public void ACorruptAppendedPayloadReadsAsNothing()
    {
        var dir = TestPaths.CreateTempDirectory();
        var exe = AppendedExe(dir, Encoding.ASCII.GetBytes("this is not a zip archive"));

        Assert.Null(PayloadExtractor.ReadEntryText(exe, DependencyInstaller.PythonRequirementEntry));
    }

    [Theory]
    [InlineData("64-bit (x64) Python 3.13", "64-bit (x64) Python 3.13")]
    [InlineData("  ARM64 Python 3.14\r\n", "ARM64 Python 3.14")]
    [InlineData("32-bit Python 3.12\n", "32-bit Python 3.12")]
    [InlineData("", null)]
    [InlineData(null, null)]
    [InlineData("Python 3.13", null)]
    [InlineData("64-bit (x64) Python 3", null)]
    [InlineData("64-bit (x64) Python 3.13\nsecond line", null)]
    [InlineData("<b>64-bit</b> Python 3.13", null)]
    public void OnlyAPythonOrgInstallerNameIsTrusted(string? text, string? expected)
    {
        Assert.Equal(expected, DependencyInstaller.ParsePythonRequirement(text));
    }

    [Fact]
    public void TheOptionsTextNamesThePythonTheServerReported()
    {
        var caption = DependencyInstaller.OptionsCaption(Requirement);
        var help = DependencyInstaller.OptionsHelpText(Requirement);

        Assert.Contains($"edge-tts and mediapipe also need {Requirement} from python.org.", caption);
        Assert.Contains($"edge-tts and mediapipe need {Requirement} from python.org.", help);
        Assert.DoesNotContain("3.12", caption);
    }

    [Fact]
    public void WithoutARequirementTheOptionsTextPointsAtTheLog()
    {
        var caption = DependencyInstaller.OptionsCaption("");

        Assert.Contains("the Python version the OpenCut server runs, from python.org", caption);
        Assert.Contains("install log names it", caption);
        Assert.DoesNotContain(" Python 3.", caption);
    }
}
