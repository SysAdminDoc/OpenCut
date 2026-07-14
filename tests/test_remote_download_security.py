"""Security and transaction tests for remote result downloads."""

import io
import wave
from pathlib import Path

import pytest

from opencut.core import remote_process, url_safety


class _Response:
    def __init__(self, chunks, *, headers=None):
        self._chunks = iter(chunks)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def read(self, _size=-1):
        item = next(self._chunks, b"")
        if isinstance(item, BaseException):
            raise item
        return item

    def geturl(self):
        return "https://node.example/api/remote/download-result/job-1"


def _configure_download(monkeypatch, root: Path, response: _Response) -> None:
    monkeypatch.setenv("OPENCUT_OUTPUT_DIR", str(root))
    monkeypatch.setattr(url_safety, "open_validated_url", lambda *_a, **_kw: response)


def _assert_no_partial(root: Path) -> None:
    assert not list(root.rglob("*.part"))


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return buffer.getvalue()


def test_remote_result_download_promotes_valid_media_atomically(tmp_path, monkeypatch):
    root = tmp_path / "output"
    media = _wav_bytes()
    response = _Response(
        [media, b""],
        headers={"Content-Length": str(len(media)), "Content-Type": "audio/wav"},
    )
    _configure_download(monkeypatch, root, response)

    result = remote_process.download_result(
        "https://node.example", "job-1", "nested/result.wav"
    )

    expected = root / "nested" / "result.wav"
    assert Path(result) == expected.resolve()
    assert expected.read_bytes() == media
    _assert_no_partial(root)


@pytest.mark.parametrize("requested", ["../escape.mp4", "nested/../../escape.mp4"])
def test_remote_result_download_rejects_traversal_before_network(
    tmp_path, monkeypatch, requested
):
    root = tmp_path / "output"
    opened = False

    def _unexpected_open(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("network must not be reached")

    monkeypatch.setenv("OPENCUT_OUTPUT_DIR", str(root))
    monkeypatch.setattr(url_safety, "open_validated_url", _unexpected_open)

    with pytest.raises(PermissionError, match="approved output directory"):
        remote_process.download_result(
            "https://node.example", "job-1", requested
        )

    assert opened is False
    assert not (tmp_path / "escape.mp4").exists()


def test_remote_result_download_rejects_absolute_path_outside_root(tmp_path, monkeypatch):
    root = tmp_path / "output"
    outside = tmp_path / "outside.mp4"
    monkeypatch.setenv("OPENCUT_OUTPUT_DIR", str(root))
    monkeypatch.setattr(
        remote_process,
        "transactional_download",
        lambda *_a, **_kw: pytest.fail("network must not be reached"),
    )

    with pytest.raises(PermissionError, match="approved output directory"):
        remote_process.download_result(
            "https://node.example", "job-1", str(outside)
        )

    assert not outside.exists()


def test_remote_result_download_rejects_path_shaped_job_id(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCUT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        remote_process,
        "transactional_download",
        lambda *_a, **_kw: pytest.fail("network must not be reached"),
    )

    with pytest.raises(PermissionError, match="Invalid remote job ID"):
        remote_process.download_result(
            "https://node.example", "../admin", "result.mp4"
        )


def test_declared_oversize_preserves_existing_result(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    final = root / "result.mp4"
    final.write_bytes(b"existing")
    response = _Response(
        [b"oversize", b""],
        headers={"Content-Length": "8", "Content-Type": "video/mp4"},
    )
    _configure_download(monkeypatch, root, response)
    monkeypatch.setattr(remote_process, "MAX_REMOTE_RESULT_BYTES", 4)
    monkeypatch.setattr(remote_process, "validate_media_download", lambda *_a: True)

    with pytest.raises(url_safety.DownloadTooLarge, match="declared size"):
        remote_process.download_result(
            "https://node.example", "job-1", str(final)
        )

    assert final.read_bytes() == b"existing"
    _assert_no_partial(root)


def test_chunked_oversize_leaves_no_result_or_partial(tmp_path, monkeypatch):
    root = tmp_path / "output"
    response = _Response(
        [b"123", b"45", b""], headers={"Content-Type": "video/mp4"}
    )
    _configure_download(monkeypatch, root, response)
    monkeypatch.setattr(remote_process, "MAX_REMOTE_RESULT_BYTES", 4)
    monkeypatch.setattr(remote_process, "validate_media_download", lambda *_a: True)

    with pytest.raises(url_safety.DownloadTooLarge, match="exceeded"):
        remote_process.download_result(
            "https://node.example", "job-1", "result.mp4"
        )

    assert not (root / "result.mp4").exists()
    _assert_no_partial(root)


def test_interrupted_download_leaves_no_result_or_partial(tmp_path, monkeypatch):
    root = tmp_path / "output"
    response = _Response(
        [b"partial", OSError("connection reset")],
        headers={"Content-Type": "video/mp4"},
    )
    _configure_download(monkeypatch, root, response)
    monkeypatch.setattr(remote_process, "validate_media_download", lambda *_a: True)

    with pytest.raises(OSError, match="connection reset"):
        remote_process.download_result(
            "https://node.example", "job-1", "result.mp4"
        )

    assert not (root / "result.mp4").exists()
    _assert_no_partial(root)


@pytest.mark.parametrize(
    ("content_type", "validator"),
    [
        ("text/html", lambda *_a: True),
        (
            "video/mp4",
            lambda *_a: (_ for _ in ()).throw(
                url_safety.DownloadContentError("not recognized media")
            ),
        ),
    ],
)
def test_mismatched_remote_content_preserves_existing_result(
    tmp_path, monkeypatch, content_type, validator
):
    root = tmp_path / "output"
    root.mkdir()
    final = root / "result.mp4"
    final.write_bytes(b"existing")
    response = _Response(
        [b"not-media", b""], headers={"Content-Type": content_type}
    )
    _configure_download(monkeypatch, root, response)
    monkeypatch.setattr(remote_process, "validate_media_download", validator)

    with pytest.raises(url_safety.DownloadContentError):
        remote_process.download_result(
            "https://node.example", "job-1", str(final)
        )

    assert final.read_bytes() == b"existing"
    _assert_no_partial(root)
