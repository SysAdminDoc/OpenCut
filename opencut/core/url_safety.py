"""Shared URL validation helpers for outbound network calls."""

import ipaddress
import json
import os
import socket
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

# Default ceiling for streamed downloads (bytes). Generous enough for large
# source media, bounded enough to stop an unbounded/chunked response from
# filling the disk. Override per-call.
DEFAULT_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB


class DownloadTooLarge(ValueError):
    """Raised when a streamed download exceeds its byte ceiling."""


class DownloadContentError(ValueError):
    """Raised when a downloaded body does not match its expected type."""


@dataclass(frozen=True)
class DownloadResult:
    """Evidence returned after a transactional download is promoted."""

    path: str
    bytes_written: int
    content_type: str
    final_url: str


def _blocked_ip_reason(addr: ipaddress._BaseAddress) -> str:
    """Return the rejection reason for an IP literal, or an empty string."""
    if addr.is_loopback or addr.is_unspecified:
        return "localhost"
    if addr.is_private or addr.is_link_local or addr.is_reserved or addr.is_multicast:
        return "private/reserved networks"
    return ""


def _validate_resolved_host(hostname: str, *, label: str) -> None:
    """Resolve *hostname* and reject if any resolved IP is local/private.

    This is the connect-time half of the SSRF guard: the structural check in
    ``validate_public_http_url`` cannot catch a public-looking hostname that
    resolves (or rebinds) to a private/loopback address. Callers that actually
    open a connection should run this immediately before connecting so the DNS
    answer they validate is the one the request will use.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ValueError(f"{label} host {hostname!r} could not be resolved") from exc

    for info in infos:
        raw_ip = info[4][0].split("%", 1)[0]  # strip IPv6 scope id
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        reason = _blocked_ip_reason(addr)
        if reason == "localhost":
            raise ValueError(f"{label} resolves to localhost ({raw_ip})")
        if reason:
            raise ValueError(f"{label} resolves to {reason} ({raw_ip})")


def validate_public_http_url(url: str, *, label: str = "URL", resolve: bool = False) -> str:
    """Validate that *url* is an HTTP(S) URL outside local/private networks.

    This is a defensive SSRF guard for user-configured outbound calls such as
    webhooks and plugin downloads. It blocks obvious local targets using only
    the URL structure itself: localhost-style names and literal IPs in
    loopback/private/reserved ranges are rejected, while regular hostnames are
    accepted without DNS lookups. That keeps validation deterministic and
    side-effect free in offline and test environments.
    """
    if not isinstance(url, str):
        raise ValueError(f"{label} is required")

    cleaned = url.strip()
    if not cleaned:
        raise ValueError(f"{label} is required")
    if any(ch in cleaned for ch in ("\r", "\n", "\x00")):
        raise ValueError(f"{label} contains invalid characters")

    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must use http:// or https:// and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not include embedded credentials")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL") from exc

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError(f"{label} has no hostname")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".localdomain"):
        raise ValueError(f"{label} must not target localhost")

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # Not a canonical IP literal. Before treating it as a regular hostname,
        # catch the alternate IPv4 encodings that ``ipaddress`` rejects but the
        # OS resolver and most HTTP clients still expand to a real address:
        # decimal (``2130706433``), octal (``0177.0.0.1``), hex (``0x7f.0.0.1``)
        # and short forms (``127.1``). ``socket.inet_aton`` performs exactly
        # this expansion offline, so the SSRF guard stays deterministic and
        # network-free while no longer being bypassable by a numeric literal.
        try:
            packed = socket.inet_aton(hostname)
        except OSError:
            # Genuine DNS hostname. The structural check cannot see where it
            # points, so callers that open a connection pass resolve=True to
            # resolve now and reject private/loopback answers (partial DNS
            # rebinding mitigation). Structure-only callers (default) accept it.
            if resolve:
                _validate_resolved_host(hostname, label=label)
            return cleaned
        addr = ipaddress.ip_address(packed)

    reason = _blocked_ip_reason(addr)
    if reason == "localhost":
        raise ValueError(f"{label} must not target localhost")
    if reason:
        raise ValueError(f"{label} must not target {reason}")

    return cleaned


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop so a public URL cannot 302 into a
    private/loopback target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_http_url(newurl, label="redirect target", resolve=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ResolvingHTTPHandler(urllib.request.HTTPHandler):
    """Re-resolve immediately before urllib opens each plain HTTP request."""

    def http_open(self, req):
        validate_public_http_url(req.full_url, label="download URL", resolve=True)
        return super().http_open(req)


class _ResolvingHTTPSHandler(urllib.request.HTTPSHandler):
    """Re-resolve immediately before urllib opens each HTTPS request."""

    def https_open(self, req):
        validate_public_http_url(req.full_url, label="download URL", resolve=True)
        return super().https_open(req)


def build_guarded_opener() -> urllib.request.OpenerDirector:
    """Return a urllib opener that re-validates every redirect hop.

    Use for outbound requests that carry a body/custom method (e.g. webhook
    POSTs) where :func:`open_validated_url` does not fit; validate the initial
    URL with ``validate_public_http_url(..., resolve=True)`` before opening.
    """
    return urllib.request.build_opener(
        _SafeRedirectHandler(), _ResolvingHTTPHandler(), _ResolvingHTTPSHandler()
    )


def open_validated_url(
    url: str,
    *,
    label: str = "URL",
    timeout: float = 120,
    headers: dict | None = None,
    resolve: bool = True,
):
    """Open *url* for reading through the SSRF guard.

    Validates the URL structurally, resolves and checks the host (when
    ``resolve``), and follows redirects only through hops that pass the same
    guard. Returns the ``http.client.HTTPResponse`` (a context manager). Callers
    remain responsible for enforcing a size ceiling while streaming the body
    (see ``stream_to_file``).
    """
    cleaned = validate_public_http_url(url, label=label, resolve=resolve)
    opener = build_guarded_opener()
    request = urllib.request.Request(cleaned, headers=headers or {})
    return opener.open(request, timeout=timeout)


def stream_to_file(response, file_obj, *, max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
                   chunk_size: int = 256 * 1024, on_chunk=None) -> int:
    """Stream *response* into *file_obj*, aborting past *max_bytes*.

    Returns the number of bytes written. Raises :class:`DownloadTooLarge` if the
    body exceeds ``max_bytes`` (checked against ``Content-Length`` up front when
    present, and enforced during the read loop for chunked/unbounded bodies).
    """
    declared = 0
    try:
        declared = int(response.headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared and declared > max_bytes:
        raise DownloadTooLarge(
            f"declared size {declared} bytes exceeds the {max_bytes}-byte ceiling"
        )

    written = 0
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        written += len(chunk)
        if written > max_bytes:
            raise DownloadTooLarge(
                f"download exceeded the {max_bytes}-byte ceiling"
            )
        file_obj.write(chunk)
        if on_chunk is not None:
            on_chunk(written, declared)
    return written


def read_response_bytes(response, *, max_bytes: int) -> bytes:
    """Read a bounded metadata response using the same length/chunk checks."""
    chunks = []

    class _Collector:
        def write(self, chunk):
            chunks.append(chunk)

    stream_to_file(response, _Collector(), max_bytes=max_bytes, chunk_size=64 * 1024)
    return b"".join(chunks)


def transactional_download(
    url: str,
    destination: str,
    *,
    max_bytes: int,
    timeout: float,
    validator: Callable[[str, object], bool | None],
    allowed_content_types: tuple[str, ...] = (),
    headers: dict | None = None,
    label: str = "download",
    local_alternative: str = "",
    on_chunk: Callable[[int, int], None] | None = None,
) -> DownloadResult:
    """Guard, stage, validate, and atomically promote one download.

    The destination is never opened directly. Existing files survive network,
    size, interruption, and validation failures.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    from opencut.config import require_network_allowed

    require_network_allowed(label, local_alternative)
    final = Path(destination).expanduser().resolve(strict=False)
    if not final.parent.is_dir():
        raise FileNotFoundError(f"Download directory does not exist: {final.parent}")

    descriptor, staged = tempfile.mkstemp(
        prefix=f".{final.name}.", suffix=".part", dir=str(final.parent)
    )
    os.close(descriptor)
    try:
        with open_validated_url(
            url,
            label=f"{label} URL",
            timeout=timeout,
            headers=headers or {"User-Agent": "OpenCut/1.0"},
        ) as response:
            content_type = str(response.headers.get("Content-Type") or "")
            media_type = content_type.split(";", 1)[0].strip().lower()
            if allowed_content_types and media_type and not any(
                media_type == allowed or media_type.startswith(allowed)
                for allowed in allowed_content_types
            ):
                raise DownloadContentError(
                    f"{label} returned unexpected Content-Type {content_type!r}"
                )
            with open(staged, "wb") as file_obj:
                written = stream_to_file(
                    response, file_obj, max_bytes=max_bytes, on_chunk=on_chunk
                )
                file_obj.flush()
                os.fsync(file_obj.fileno())
            if written <= 0:
                raise DownloadContentError(f"{label} returned an empty body")
            if validator(staged, response) is False:
                raise DownloadContentError(f"{label} failed content validation")
            final_url = response.geturl() if hasattr(response, "geturl") else url
        os.replace(staged, final)
        return DownloadResult(str(final), written, content_type, str(final_url))
    finally:
        try:
            os.unlink(staged)
        except FileNotFoundError:
            pass


def validate_media_download(path: str, _response=None) -> bool:
    """Require ffprobe to recognize at least one audio/video stream."""
    from opencut.helpers import get_ffprobe_path

    command = [
        get_ffprobe_path(),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DownloadContentError(f"media validation unavailable: {exc}") from exc
    if result.returncode != 0:
        raise DownloadContentError("download is not recognized media")
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise DownloadContentError("ffprobe returned invalid media metadata") from exc
    if not any(stream.get("codec_type") in {"audio", "video"} for stream in streams):
        raise DownloadContentError("download contains no audio/video stream")
    return True


def validate_zip_download(path: str, _response=None) -> bool:
    if not zipfile.is_zipfile(path):
        raise DownloadContentError("download is not a ZIP archive")
    return True


def validate_font_download(path: str, _response=None) -> bool:
    """Recognize TTF/OTF/WOFF/WOFF2 container signatures."""
    with open(path, "rb") as handle:
        signature = handle.read(4)
    if signature not in {b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1", b"wOFF", b"wOF2"}:
        raise DownloadContentError("download is not a recognized font file")
    return True


def validate_pytorch_weights(path: str, _response=None) -> bool:
    """Reject HTML/text masquerading as PyTorch ZIP/pickle weights."""
    if os.path.getsize(path) < 1024:
        raise DownloadContentError("model weights are unexpectedly small")
    with open(path, "rb") as handle:
        signature = handle.read(4)
    if not (signature.startswith(b"PK") or signature[:1] == b"\x80"):
        raise DownloadContentError("download is not a PyTorch weight archive")
    return True
