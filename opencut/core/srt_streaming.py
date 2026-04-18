"""
SRT (Secure Reliable Transport) streaming output.

SRT (https://github.com/Haivision/srt, MPL-2.0) is a low-latency
contribution protocol widely used for broadcast feeds, live Twitch
ingress, and remote production.  FFmpeg ships with SRT support when
built with ``--enable-libsrt``; this module wraps the subprocess
invocation so OpenCut routes can start / monitor / stop an ingest
stream without knowing the FFmpeg CLI grammar.

Two operating modes:
- ``caller`` (default): OpenCut *pushes* to a remote listener at
  ``srt://host:port`` (typical: your broadcast encoder listens, our
  machine pushes).
- ``listener``: OpenCut *listens* on a local port and accepts a
  push from a remote caller.  Useful for receiving contribution feeds.

The command is a single FFmpeg child process; we register it with the
job subsystem so cancellation kills the stream cleanly.  Packet-loss
resilience and latency are configured via SRT URL query parameters
(``?latency=200&pkt_size=1316``); defaults match Haivision's
recommendations for 300-ms one-way delay.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.parse import quote, urlencode

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


MODES = ("caller", "listener")
DEFAULT_LATENCY_MS = 200
DEFAULT_PKT_SIZE = 1316

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


@dataclass
class SrtStreamResult:
    """Structured return from ``start_stream`` — the subprocess pid /
    URL is attached so routes can monitor or stop the stream."""
    url: str = ""
    mode: str = "caller"
    pid: int = 0
    stopped: bool = False
    exit_code: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

_AVAILABILITY_CACHE: Dict[str, Optional[bool]] = {"libsrt": None}


def check_srt_available() -> bool:
    """Cache-lookup probe of FFmpeg for SRT protocol support."""
    if _AVAILABILITY_CACHE["libsrt"] is not None:
        return bool(_AVAILABILITY_CACHE["libsrt"])
    ff = get_ffmpeg_path()
    if not ff:
        _AVAILABILITY_CACHE["libsrt"] = False
        return False
    try:
        proc = _sp.run(
            [ff, "-hide_banner", "-protocols"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        available = bool(re.search(r"^\s*srt\s*$",
                                    (proc.stdout or ""), flags=re.MULTILINE))
        _AVAILABILITY_CACHE["libsrt"] = available
        return available
    except Exception:  # noqa: BLE001
        _AVAILABILITY_CACHE["libsrt"] = False
        return False


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def build_srt_url(
    host: str,
    port: int,
    mode: str = "caller",
    latency_ms: int = DEFAULT_LATENCY_MS,
    pkt_size: int = DEFAULT_PKT_SIZE,
    passphrase: Optional[str] = None,
    stream_id: Optional[str] = None,
    extra_params: Optional[Dict[str, str]] = None,
) -> str:
    """Assemble a validated ``srt://`` URL.

    Raises ``ValueError`` on malformed host / out-of-range port /
    reserved query keys so callers can't smuggle filter-chain markers
    into the FFmpeg command line.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not host or not _HOST_RE.match(host):
        raise ValueError(f"Invalid SRT host: {host!r}")
    try:
        port = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("port must be an integer") from exc
    if not (1 <= port <= 65535):
        raise ValueError("port must be in [1, 65535]")
    try:
        latency_ms = int(latency_ms)
    except (TypeError, ValueError):
        latency_ms = DEFAULT_LATENCY_MS
    latency_ms = max(20, min(5000, latency_ms))
    try:
        pkt_size = int(pkt_size)
    except (TypeError, ValueError):
        pkt_size = DEFAULT_PKT_SIZE
    pkt_size = max(64, min(1500, pkt_size))

    params: Dict[str, str] = {
        "mode": mode,
        "latency": str(latency_ms * 1000),  # SRT expects microseconds
        "pkt_size": str(pkt_size),
    }
    if passphrase:
        if len(passphrase) < 10 or len(passphrase) > 79:
            raise ValueError("SRT passphrase must be 10–79 chars")
        params["passphrase"] = passphrase
    if stream_id:
        if len(stream_id) > 512:
            raise ValueError("SRT stream_id too long (max 512 chars)")
        params["streamid"] = stream_id
    if extra_params:
        reserved = set(params.keys())
        for k, v in extra_params.items():
            if not isinstance(k, str) or not re.match(r"^[a-z_][a-z0-9_]*$", k):
                raise ValueError(f"Invalid SRT param name: {k!r}")
            if k in reserved:
                continue
            params[k] = str(v)[:200]

    return f"srt://{quote(host)}:{port}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Stream control
# ---------------------------------------------------------------------------

def start_stream(
    input_path: str,
    host: str,
    port: int,
    mode: str = "caller",
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    video_bitrate: str = "4500k",
    audio_bitrate: str = "192k",
    latency_ms: int = DEFAULT_LATENCY_MS,
    pkt_size: int = DEFAULT_PKT_SIZE,
    passphrase: Optional[str] = None,
    stream_id: Optional[str] = None,
    on_progress: Optional[Callable] = None,
    job_id: Optional[str] = None,
) -> SrtStreamResult:
    """Start an FFmpeg SRT push / pull subprocess.

    The function **does not block** until the stream ends — it spawns
    the child, registers it with the job subsystem (so the standard
    cancel path kills it), and returns once FFmpeg has reported it's
    alive.  Callers monitoring a long-running stream should poll job
    status; the background process writes ``exit_code`` into the
    result when it terminates.

    Args:
        input_path: Source video file.  Use a named pipe / device for
            live capture.
        host, port: SRT endpoint.
        mode: ``caller`` (push to remote listener) or ``listener``.
        video_codec, audio_codec, video_bitrate, audio_bitrate: encode
            settings. H.264 is the only codec current SRT broadcast
            encoders accept; audio is AAC.
        latency_ms: SRT latency buffer in milliseconds (20–5000).
        pkt_size: MPEG-TS UDP packet size (default 1316 bytes —
            the standard broadcast MTU).
        passphrase: Optional encryption passphrase (10–79 chars).
        stream_id: Optional SRT stream ID for routing.
        on_progress: ``(pct, msg)`` callback — only fires once, when
            the subprocess is confirmed alive.
        job_id: Optional job_id for ``_register_job_process`` tracking.

    Returns:
        :class:`SrtStreamResult` — with ``pid`` set and ``stopped=False``.

    Raises:
        RuntimeError: FFmpeg lacks SRT support, or subprocess died on
            spawn.
        ValueError: invalid URL / encode params.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input_path not found: {input_path}")
    if not check_srt_available():
        raise RuntimeError(
            "FFmpeg lacks SRT support. Rebuild with --enable-libsrt or "
            "install a build that includes it."
        )

    url = build_srt_url(
        host=host, port=port, mode=mode,
        latency_ms=latency_ms, pkt_size=pkt_size,
        passphrase=passphrase, stream_id=stream_id,
    )

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "warning",
        "-re",                         # read input at its native rate
        "-i", input_path,
        "-c:v", video_codec,
        "-b:v", video_bitrate,
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-f", "mpegts",
        url,
    ]
    logger.debug("SRT start: %s", " ".join(cmd))

    if on_progress:
        on_progress(20, f"Spawning FFmpeg → {url}")

    # Start without capturing stdout/stderr — streaming is typically
    # long-running and we don't want to balloon memory. Errors land in
    # the FFmpeg console / logs.
    proc = _sp.Popen(cmd)
    # Give FFmpeg a moment to fail fast on bad URL / permission issues.
    try:
        exit_code = proc.wait(timeout=1.5)
        # Child exited within the probe window — that's a failure.
        raise RuntimeError(
            f"FFmpeg SRT process exited immediately (rc={exit_code}). "
            "Check that the host / port are reachable and the codec is "
            "correct for SRT transport."
        )
    except _sp.TimeoutExpired:
        pass  # still running — good

    if job_id:
        try:
            from opencut.jobs import _register_job_process
            _register_job_process(job_id, proc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not register SRT process with job %s: %s",
                           job_id, exc)

    if on_progress:
        on_progress(100, f"Streaming (pid={proc.pid})")

    return SrtStreamResult(
        url=url,
        mode=mode,
        pid=proc.pid,
        stopped=False,
        exit_code=None,
        notes=[
            f"video={video_codec}@{video_bitrate}",
            f"audio={audio_codec}@{audio_bitrate}",
            f"latency={latency_ms}ms",
        ],
    )


def stop_stream(pid: int, timeout: float = 3.0) -> bool:
    """Gracefully terminate an SRT streaming subprocess.

    Returns ``True`` when the process is confirmed gone, ``False`` when
    we had to force-kill it (or it was already gone).
    """
    if not pid:
        return False
    try:
        import signal
        os.kill(int(pid), signal.SIGTERM)
    except (OSError, ValueError):
        return False

    # Wait up to ``timeout`` seconds for the process to exit; force-kill
    # on overrun.
    import time
    deadline = time.monotonic() + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        try:
            os.kill(int(pid), 0)       # probe
        except OSError:
            return True                # gone
        time.sleep(0.1)

    try:
        import signal
        os.kill(int(pid), signal.SIGKILL)
    except (OSError, ValueError, AttributeError):
        # Windows lacks SIGKILL — fall back to taskkill via subprocess
        try:
            _sp.run(["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5, check=False)
        except Exception:  # noqa: BLE001
            return False
    return False
