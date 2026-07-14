"""SSRF, size-ceiling, local-only, and media-verification guards for URL ingest.

Covers the P0 fix that routed opencut.core.url_ingest through the shared
connect-time SSRF guard in opencut.core.url_safety, plus the reusable
redirect/resolve/size helpers used by webhooks.
"""

import io
import os

import pytest

from opencut.core import url_ingest, url_safety
from opencut.core.url_safety import (
    DownloadTooLarge,
    _SafeRedirectHandler,
    stream_to_file,
    validate_public_http_url,
)


class _FakeResponse:
    def __init__(self, chunks, content_length=None):
        self._chunks = list(chunks)
        headers = {}
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        self.headers = headers

    def read(self, size=-1):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


# --- structural SSRF guard -------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://localhost:5679/health",
        "http://[::1]/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/x",
        "http://192.168.1.1/x",
        "http://2130706433/",  # decimal-encoded 127.0.0.1
        "http://0x7f.0.0.1/",  # hex-encoded
    ],
)
def test_validate_rejects_local_and_private_targets(url):
    with pytest.raises(ValueError):
        validate_public_http_url(url)


def test_validate_accepts_public_literal():
    assert validate_public_http_url("http://93.184.216.34/x") == "http://93.184.216.34/x"


# --- connect-time resolved-IP guard (partial rebinding mitigation) ---------

def test_resolve_rejects_hostname_pointing_private(monkeypatch):
    def fake_getaddrinfo(host, *a, **k):
        return [(2, 1, 6, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError):
        validate_public_http_url("http://rebind.example/x", resolve=True)


def test_resolve_accepts_hostname_pointing_public(monkeypatch):
    def fake_getaddrinfo(host, *a, **k):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)
    assert validate_public_http_url("http://ok.example/x", resolve=True)


def test_resolve_is_opt_in(monkeypatch):
    # Default (resolve=False) must not touch the network.
    def boom(*a, **k):
        raise AssertionError("getaddrinfo must not be called when resolve=False")

    monkeypatch.setattr(url_safety.socket, "getaddrinfo", boom)
    assert validate_public_http_url("http://public.example/x")


# --- redirect re-validation ------------------------------------------------

def test_redirect_handler_rejects_private_hop():
    handler = _SafeRedirectHandler()
    with pytest.raises(ValueError):
        handler.redirect_request(
            object(), None, 302, "Found", {}, "http://127.0.0.1/internal"
        )


# --- size ceiling ----------------------------------------------------------

def test_stream_to_file_enforces_ceiling_during_read():
    resp = _FakeResponse([b"a" * 100, b"b" * 100, b"c" * 100])
    with pytest.raises(DownloadTooLarge):
        stream_to_file(resp, io.BytesIO(), max_bytes=150, chunk_size=100)


def test_stream_to_file_rejects_declared_oversize():
    resp = _FakeResponse([b"a"], content_length=10_000)
    with pytest.raises(DownloadTooLarge):
        stream_to_file(resp, io.BytesIO(), max_bytes=1000)


def test_stream_to_file_writes_within_ceiling():
    resp = _FakeResponse([b"abc", b"def"])
    out = io.BytesIO()
    written = stream_to_file(resp, out, max_bytes=100)
    assert written == 6
    assert out.getvalue() == b"abcdef"


# --- ingest entry point ----------------------------------------------------

def test_ingest_url_blocked_in_local_only(monkeypatch):
    monkeypatch.setenv("OPENCUT_LOCAL_ONLY", "1")
    with pytest.raises(RuntimeError):
        url_ingest.ingest_url("http://93.184.216.34/video.mp4")


def test_ingest_url_rejects_loopback(monkeypatch):
    # Force local-only off so the SSRF check (not the network gate) is exercised.
    monkeypatch.setattr("opencut.config.is_local_only", lambda: False)
    with pytest.raises(ValueError):
        url_ingest.ingest_url("http://127.0.0.1:5679/steal")


def test_ingest_url_requires_scheme(monkeypatch):
    monkeypatch.setattr("opencut.config.is_local_only", lambda: False)
    with pytest.raises(ValueError):
        url_ingest.ingest_url("ftp://example.com/x")


# --- media verification ----------------------------------------------------

def test_verify_is_media_rejects_non_media(tmp_path):
    from opencut.helpers import get_ffprobe_path

    probe = get_ffprobe_path()
    if not probe or not os.path.exists(probe):
        pytest.skip("ffprobe unavailable")
    fake = tmp_path / "notavideo.mp4"
    fake.write_text("<html>this is not a video</html>")
    assert url_ingest._verify_is_media(str(fake)) is False


def test_max_ingest_bytes_env_override(monkeypatch):
    monkeypatch.setenv("OPENCUT_MAX_INGEST_BYTES", "12345")
    assert url_ingest._max_ingest_bytes() == 12345
    monkeypatch.setenv("OPENCUT_MAX_INGEST_BYTES", "garbage")
    assert url_ingest._max_ingest_bytes() == url_safety.DEFAULT_MAX_DOWNLOAD_BYTES
