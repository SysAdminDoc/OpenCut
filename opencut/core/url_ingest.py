"""Import-by-link footage ingest.

Fetches video from a URL to the local media cache so existing
OpenCut routes can consume it as a local file path.  Uses yt-dlp
when installed (handles YouTube, Vimeo, Zoom Clips, Medal, etc.);
falls back to a plain HTTP download for direct-link URLs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("opencut")

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "media_ingest")

INSTALL_HINT = "pip install yt-dlp (optional; direct URLs work without it)"


def _max_ingest_bytes() -> int:
    """Byte ceiling for a single direct download (env-overridable)."""
    from .url_safety import DEFAULT_MAX_DOWNLOAD_BYTES

    raw = os.environ.get("OPENCUT_MAX_INGEST_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid OPENCUT_MAX_INGEST_BYTES=%r — using default", raw)
    return DEFAULT_MAX_DOWNLOAD_BYTES


def _verify_is_media(path: str) -> bool:
    """Return True if ffprobe recognizes at least one media stream in *path*.

    The direct-download path derives its filename/extension from the (untrusted)
    URL and never inspects the bytes, so an attacker-controlled endpoint can
    serve HTML/JSON under a ``.mp4`` name. Rejecting non-media here keeps the
    corrupt file out of the cache and out of the rest of the pipeline.
    """
    from opencut.helpers import get_ffprobe_path

    cmd = [
        get_ffprobe_path(), "-v", "error",
        "-show_entries", "stream=codec_type",
        "-of", "json", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        # ffprobe unavailable — do not block ingest solely because the probe
        # tool is missing; the SSRF/size guards already ran.
        logger.warning("ffprobe unavailable for media verification: %s", exc)
        return True
    if proc.returncode != 0:
        return False
    import json as _json

    try:
        streams = _json.loads(proc.stdout.decode(errors="replace")).get("streams", [])
    except (ValueError, AttributeError):
        return False
    return bool(streams)


@dataclass
class IngestResult:
    filepath: str = ""
    url: str = ""
    title: str = ""
    duration: float = 0.0
    filesize_mb: float = 0.0
    source: str = ""
    cached: bool = False
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return (
            "filepath", "url", "title", "duration",
            "filesize_mb", "source", "cached", "notes",
        )


def check_ytdlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _cached_path(url: str) -> Optional[str]:
    key = _cache_key(url)
    if not os.path.isdir(CACHE_DIR):
        return None
    for entry in os.listdir(CACHE_DIR):
        if entry.startswith(key):
            full = os.path.join(CACHE_DIR, entry)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                return full
    return None


def _fetch_ytdlp(
    url: str,
    output_dir: str,
    on_progress: Optional[Callable] = None,
) -> IngestResult:
    key = _cache_key(url)
    output_template = os.path.join(output_dir, f"{key}_%(title).80B.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--print-json",
        "--no-simulate",
        url,
    ]

    if on_progress:
        on_progress(10, "Downloading via yt-dlp...")

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"yt-dlp failed (rc={proc.returncode}): {stderr[:500]}")

    import json
    info = {}
    for line in (proc.stdout or "").strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    filepath = info.get("_filename", "")
    if not filepath or not os.path.isfile(filepath):
        for entry in os.listdir(output_dir):
            if entry.startswith(key):
                filepath = os.path.join(output_dir, entry)
                break

    if not filepath or not os.path.isfile(filepath):
        raise RuntimeError("yt-dlp completed but no output file found")

    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, "Download complete")

    return IngestResult(
        filepath=filepath,
        url=url,
        title=info.get("title", os.path.basename(filepath)),
        duration=float(info.get("duration") or 0),
        filesize_mb=size_mb,
        source="yt-dlp",
        notes=[],
    )


def _fetch_direct(
    url: str,
    output_dir: str,
    on_progress: Optional[Callable] = None,
) -> IngestResult:
    key = _cache_key(url)
    ext = ".mp4"
    url_lower = url.lower().split("?")[0]
    for candidate in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts"):
        if url_lower.endswith(candidate):
            ext = candidate
            break

    output_path = os.path.join(output_dir, f"{key}{ext}")

    if on_progress:
        on_progress(10, "Downloading file...")

    from .url_safety import open_validated_url, stream_to_file

    max_bytes = _max_ingest_bytes()

    def _report(written: int, total: int) -> None:
        if on_progress and total > 0:
            pct = min(95, int(10 + 85 * written / total))
            on_progress(pct, f"Downloaded {written // (1024*1024)} MB")

    # Route through the SSRF guard: validates the URL, resolves and rejects
    # private/loopback hosts, and re-validates every redirect hop.
    with open_validated_url(
        url, label="ingest URL", timeout=120,
        headers={"User-Agent": "OpenCut/1.0"},
    ) as resp:
        _fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=output_dir)
        os.close(_fd)
        try:
            with open(tmp_path, "wb") as f:
                stream_to_file(resp, f, max_bytes=max_bytes, on_chunk=_report)
            if not _verify_is_media(tmp_path):
                raise ValueError(
                    "downloaded content is not a recognized media file "
                    "(the URL did not return video/audio)"
                )
            os.replace(tmp_path, output_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, "Download complete")

    return IngestResult(
        filepath=output_path,
        url=url,
        title=os.path.basename(output_path),
        duration=0.0,
        filesize_mb=size_mb,
        source="direct",
        notes=["Duration unknown — run through ffprobe for metadata"],
    )


def ingest_url(
    url: str,
    on_progress: Optional[Callable] = None,
) -> IngestResult:
    """Fetch a URL to the local media cache.

    Uses yt-dlp when installed (YouTube, Vimeo, Zoom, Medal, etc.)
    and falls back to a plain HTTP download for direct-link URLs.
    Returns an IngestResult with the local filepath ready for any
    existing OpenCut route.
    """
    if not url or not url.strip():
        raise ValueError("url is required")
    url = url.strip()

    # Cheap scheme check first so malformed input is rejected consistently even
    # when local-only mode is active.
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must be http:// or https://")

    # Local-only mode disables every network egress feature; URL ingest is one.
    from opencut.config import require_network_allowed

    require_network_allowed(
        "URL ingest", local_alternative="a local file path or the media browser"
    )

    # Structural SSRF pre-check (credentials, literal private/loopback IPs,
    # numeric-encoding bypasses). The connect-time resolved-IP and redirect
    # checks run again inside the direct-download path.
    from .url_safety import validate_public_http_url

    url = validate_public_http_url(url, label="ingest URL")

    cached = _cached_path(url)
    if cached:
        size_mb = round(os.path.getsize(cached) / (1024 * 1024), 2)
        if on_progress:
            on_progress(100, "Using cached file")
        return IngestResult(
            filepath=cached,
            url=url,
            title=os.path.basename(cached),
            filesize_mb=size_mb,
            source="cache",
            cached=True,
        )

    os.makedirs(CACHE_DIR, exist_ok=True)

    if check_ytdlp_available():
        return _fetch_ytdlp(url, CACHE_DIR, on_progress)

    return _fetch_direct(url, CACHE_DIR, on_progress)


def list_cached() -> list:
    """Return metadata for all cached ingest files."""
    if not os.path.isdir(CACHE_DIR):
        return []
    entries = []
    for name in sorted(os.listdir(CACHE_DIR)):
        full = os.path.join(CACHE_DIR, name)
        if not os.path.isfile(full):
            continue
        entries.append({
            "filename": name,
            "filepath": full,
            "size_mb": round(os.path.getsize(full) / (1024 * 1024), 2),
        })
    return entries


def clear_cache() -> int:
    """Remove all cached ingest files. Returns count of files removed."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    count = 0
    for name in os.listdir(CACHE_DIR):
        full = os.path.join(CACHE_DIR, name)
        if os.path.isfile(full):
            os.unlink(full)
            count += 1
    return count
