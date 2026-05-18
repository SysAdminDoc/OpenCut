using System.Text;

namespace OpenCut.Installer.Tests;

public class SubStreamTests
{
    [Fact]
    public void ReadIsConstrainedToConfiguredWindow()
    {
        using var backing = new MemoryStream(Encoding.ASCII.GetBytes("0123456789"));
        using var stream = new SubStream(backing, offset: 2, length: 5);
        var buffer = new byte[10];

        var read = stream.Read(buffer, 0, buffer.Length);

        Assert.Equal(5, read);
        Assert.Equal("23456", Encoding.ASCII.GetString(buffer, 0, read));
        Assert.Equal(0, stream.Read(buffer, 0, buffer.Length));
    }

    [Fact]
    public void SeekClampsToSubStreamBounds()
    {
        using var backing = new MemoryStream(Encoding.ASCII.GetBytes("0123456789"));
        using var stream = new SubStream(backing, offset: 2, length: 5);

        Assert.Equal(5, stream.Seek(100, SeekOrigin.Begin));
        Assert.Equal(0, stream.Seek(-100, SeekOrigin.Begin));
        Assert.Equal(4, stream.Seek(-1, SeekOrigin.End));
    }

    [Fact]
    public void WriteOperationsAreRejected()
    {
        using var backing = new MemoryStream(Encoding.ASCII.GetBytes("0123456789"));
        using var stream = new SubStream(backing, offset: 2, length: 5);

        Assert.False(stream.CanWrite);
        Assert.Throws<NotSupportedException>(() => stream.SetLength(1));
        Assert.Throws<NotSupportedException>(() => stream.Write([1, 2], 0, 2));
    }
}
