using System.IO;

namespace OpenCut.Installer;

/// <summary>
/// Read-only stream wrapper that exposes a subrange of an underlying stream.
/// Used to read appended payload data from the self-extracting exe.
/// </summary>
public sealed class SubStream : Stream
{
    private readonly Stream _baseStream;
    private readonly long _offset;
    private readonly long _length;
    private long _position;

    public SubStream(Stream baseStream, long offset, long length)
    {
        _baseStream = baseStream;
        _offset = offset;
        _length = length;
        _position = 0;
        _baseStream.Seek(_offset, SeekOrigin.Begin);
    }

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => false;
    public override long Length => _length;

    public override long Position
    {
        get => _position;
        set => Seek(value, SeekOrigin.Begin);
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        long remaining = _length - _position;
        if (remaining <= 0) return 0;
        if (count > remaining) count = (int)remaining;

        _baseStream.Seek(_offset + _position, SeekOrigin.Begin);
        int read = _baseStream.Read(buffer, offset, count);
        _position += read;
        return read;
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        _position = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _position + offset,
            SeekOrigin.End => _length + offset,
            _ => throw new ArgumentException("Invalid origin", nameof(origin))
        };
        if (_position < 0) _position = 0;
        if (_position > _length) _position = _length;
        return _position;
    }

    public override void Flush() { }
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
