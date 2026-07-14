"""SSRF, cap, validation, cleanup, and atomicity tests for direct downloads."""

import json
import socket
import zipfile
from pathlib import Path

import pytest

from opencut.core import url_safety
from opencut.core.url_safety import (
    DownloadContentError,
    DownloadTooLarge,
    transactional_download,
)


class _FakeResponse:
    def __init__(self, chunks, headers=None, final_url="https://cdn.example/file"):
        self._chunks = list(chunks)
        self.headers = headers or {}
        self._final_url = final_url

    def read(self, _size=-1):
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk

    def geturl(self):
        return self._final_url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _download(monkeypatch, response, destination, **kwargs):
    monkeypatch.setattr(url_safety, "open_validated_url", lambda *_a, **_k: response)
    return transactional_download(
        "https://public.example/file",
        str(destination),
        max_bytes=kwargs.pop("max_bytes", 1024),
        timeout=5,
        validator=kwargs.pop("validator", lambda *_a: True),
        label="test download",
        **kwargs,
    )


def test_dns_rebinding_public_then_private_is_rejected_before_connection(monkeypatch):
    answers = [
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 0))],
    ]

    def resolve(*_args, **_kwargs):
        return answers.pop(0)

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="private/reserved"):
        url_safety.open_validated_url("https://rebind.example/file", timeout=1)
    assert answers == []


def test_declared_oversize_preserves_existing_destination(monkeypatch, tmp_path):
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"old")
    response = _FakeResponse([b"body"], {"Content-Length": "2048"})

    with pytest.raises(DownloadTooLarge, match="declared size"):
        _download(monkeypatch, response, destination, max_bytes=1024)

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".asset.bin.*.part")) == []


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "2"}])
def test_missing_or_false_content_length_cannot_bypass_chunk_cap(
    monkeypatch, tmp_path, headers
):
    destination = tmp_path / "asset.bin"
    response = _FakeResponse([b"1234", b"5678"], headers)

    with pytest.raises(DownloadTooLarge, match="download exceeded"):
        _download(monkeypatch, response, destination, max_bytes=6)

    assert not destination.exists()
    assert list(tmp_path.glob(".asset.bin.*.part")) == []


def test_interruption_cleans_partial_and_preserves_existing_destination(monkeypatch, tmp_path):
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"old")
    response = _FakeResponse([b"partial", OSError("connection reset")])

    with pytest.raises(OSError, match="connection reset"):
        _download(monkeypatch, response, destination)

    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".asset.bin.*.part")) == []


def test_content_type_mismatch_fails_before_promotion(monkeypatch, tmp_path):
    destination = tmp_path / "plugin.zip"
    response = _FakeResponse([b"<html>nope</html>"], {"Content-Type": "text/html"})

    with pytest.raises(DownloadContentError, match="unexpected Content-Type"):
        _download(
            monkeypatch,
            response,
            destination,
            allowed_content_types=("application/zip",),
        )

    assert not destination.exists()


def test_validator_mismatch_cleans_partial(monkeypatch, tmp_path):
    destination = tmp_path / "font.woff2"
    response = _FakeResponse([b"not-font"], {"Content-Type": "font/woff2"})

    with pytest.raises(DownloadContentError, match="recognized font"):
        _download(
            monkeypatch,
            response,
            destination,
            validator=url_safety.validate_font_download,
            allowed_content_types=("font/",),
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".font.woff2.*.part")) == []


def test_success_fsyncs_validates_and_atomically_replaces(monkeypatch, tmp_path):
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"old")
    validated = []
    response = _FakeResponse(
        [b"new-", b"content"],
        {"Content-Type": "application/octet-stream"},
        "https://cdn.example/final.bin",
    )

    def validator(path, received_response):
        validated.append((Path(path).read_bytes(), received_response.geturl()))
        return True

    result = _download(
        monkeypatch,
        response,
        destination,
        validator=validator,
        allowed_content_types=("application/octet-stream",),
    )

    assert destination.read_bytes() == b"new-content"
    assert validated == [(b"new-content", "https://cdn.example/final.bin")]
    assert result.bytes_written == 11
    assert result.final_url == "https://cdn.example/final.bin"
    assert list(tmp_path.glob(".asset.bin.*.part")) == []


def test_zip_font_and_weight_validators_reject_masquerading_html(tmp_path):
    fake = tmp_path / "fake.bin"
    fake.write_bytes(b"<html>" + b"x" * 2048)

    with pytest.raises(DownloadContentError, match="ZIP"):
        url_safety.validate_zip_download(str(fake))
    with pytest.raises(DownloadContentError, match="font"):
        url_safety.validate_font_download(str(fake))
    with pytest.raises(DownloadContentError, match="PyTorch"):
        url_safety.validate_pytorch_weights(str(fake))


def test_zip_font_and_weight_validators_accept_known_signatures(tmp_path):
    archive = tmp_path / "plugin.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("plugin.json", "{}")
    font = tmp_path / "font.woff2"
    font.write_bytes(b"wOF2" + b"font")
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"\x80\x04" + b"x" * 2048)

    assert url_safety.validate_zip_download(str(archive))
    assert url_safety.validate_font_download(str(font))
    assert url_safety.validate_pytorch_weights(str(weights))


def test_media_validator_requires_audio_or_video_stream(monkeypatch, tmp_path):
    media = tmp_path / "media.bin"
    media.write_bytes(b"bytes")

    class _Result:
        returncode = 0
        stdout = json.dumps({"streams": [{"codec_type": "subtitle"}]})

    monkeypatch.setattr(url_safety.subprocess, "run", lambda *_a, **_k: _Result())
    with pytest.raises(DownloadContentError, match="no audio/video"):
        url_safety.validate_media_download(str(media))


def test_all_reviewed_direct_download_callers_use_transactional_helper():
    root = Path(__file__).resolve().parents[1] / "opencut" / "core"
    reviewed = [
        "stock_search.py",
        "plugin_marketplace.py",
        "sfx_library.py",
        "google_fonts.py",
        "upscale_pro.py",
    ]
    for name in reviewed:
        source = (root / name).read_text(encoding="utf-8")
        assert "urlretrieve(" not in source, name
        assert "transactional_download(" in source, name
