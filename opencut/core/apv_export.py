"""APV (Advanced Professional Video) export via liboapv.

APV is Samsung's royalty-free professional intermediate codec, standardised as
IETF RFC 9924 (February 2026) and supported natively by FFmpeg 8.0+. Unlike the
delivery codecs OpenCut already exposes (AV1, VVC, H.264/5), APV is an
*all-intra mezzanine* format: every frame is independently coded, so it survives
repeated decode/encode generations without the drift that inter-coded formats
accumulate. That is exactly the round-trip an editing bridge produces, which is
why this sits alongside :mod:`opencut.core.vvc_export` rather than replacing it.

Availability is probed against the linked FFmpeg — a build without
``--enable-liboapv`` reports the encoder as absent and callers get a clear
``MISSING_DEPENDENCY`` instead of a failed job. The bundled Windows payload is
compiled with it.

Presets trade encode speed against bitrate; APV is intentionally large, so the
defaults target mezzanine use, not delivery.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

#: FFmpeg encoder name provided by ``--enable-liboapv``.
APV_ENCODER = "liboapv"

#: Containers FFmpeg can mux APV into.
APV_CONTAINERS = (".mp4", ".apv")

INSTALL_HINT = (
    "This FFmpeg build lacks the APV encoder. Use a build configured with "
    "--enable-liboapv (the bundled Windows payload has it)."
)

# ---------------------------------------------------------------------------
# Preset packs — APV is a mezzanine format, so these are quality-first.
# ---------------------------------------------------------------------------
APV_PRESETS: Dict[str, Dict[str, str]] = {
    "fast": {
        "description": "Fastest encode; largest files. On-set / offload use.",
        "preset": "fast",
        "qp": "26",
    },
    "balanced": {
        "description": "Recommended mezzanine default for edit round-trips.",
        "preset": "medium",
        "qp": "22",
    },
    "archive": {
        "description": "Visually transparent archival master; slowest.",
        "preset": "slow",
        "qp": "16",
    },
}

#: APV is 4:2:2 10-bit oriented; this is the safest widely-supported input.
DEFAULT_PIX_FMT = "yuv422p10le"


@dataclass
class ApvResult:
    """Subscriptable so route handlers can hand it straight to ``jsonify``."""

    output: str
    preset: str
    qp: int
    container: str
    pix_fmt: str
    size_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


_AVAILABILITY_CACHE: Dict[str, Optional[bool]] = {APV_ENCODER: None, "decoder": None}


def _probe(kind: str, pattern: str, cache_key: str) -> bool:
    if _AVAILABILITY_CACHE[cache_key] is not None:
        return bool(_AVAILABILITY_CACHE[cache_key])
    ff = get_ffmpeg_path()
    if not ff:
        _AVAILABILITY_CACHE[cache_key] = False
        return False
    try:
        proc = _sp.run(
            [ff, "-hide_banner", kind],
            capture_output=True, text=True, timeout=15, check=False,
        )
        present = bool(re.search(pattern, proc.stdout or "", flags=re.MULTILINE))
    except Exception:  # noqa: BLE001 - a failed probe means "not available"
        present = False
    _AVAILABILITY_CACHE[cache_key] = present
    return present


def check_apv_available() -> bool:
    """True when the linked FFmpeg exposes the ``liboapv`` encoder."""
    return _probe("-encoders", rf"^\s*\S*\s+{APV_ENCODER}\b", APV_ENCODER)


def check_apv_decode_available() -> bool:
    """True when the linked FFmpeg can decode APV."""
    return _probe("-decoders", r"^\s*\S*\s+apv\b", "decoder")


def clear_availability_cache() -> None:
    """Reset the probe cache (tests, and after swapping the FFmpeg binary)."""
    for key in _AVAILABILITY_CACHE:
        _AVAILABILITY_CACHE[key] = None


def list_presets() -> List[Dict[str, str]]:
    return [{"name": name, **spec} for name, spec in APV_PRESETS.items()]


def apv_info() -> Dict[str, object]:
    """Capability report for the info route."""
    available = check_apv_available()
    return {
        "available": available,
        "encoder": APV_ENCODER,
        "decode_available": check_apv_decode_available(),
        "standard": "IETF RFC 9924",
        "containers": list(APV_CONTAINERS),
        "default_pix_fmt": DEFAULT_PIX_FMT,
        "presets": list_presets(),
        "install_hint": None if available else INSTALL_HINT,
    }


def encode_apv(
    input_path: str,
    preset: str = "balanced",
    output: Optional[str] = None,
    qp_override: Optional[int] = None,
    container: str = ".mp4",
    pix_fmt: str = DEFAULT_PIX_FMT,
    on_progress: Optional[Callable] = None,
) -> ApvResult:
    """Encode *input_path* to APV.

    Raises ``ValueError`` for bad arguments and ``RuntimeError`` when the
    linked FFmpeg has no APV encoder, so the caller surfaces a dependency
    error rather than a failed render.
    """
    if not input_path or not os.path.isfile(input_path):
        raise ValueError(f"Input file not found: {input_path!r}")
    if preset not in APV_PRESETS:
        raise ValueError(
            f"Unknown APV preset {preset!r}. Choose one of: {', '.join(APV_PRESETS)}"
        )
    if container not in APV_CONTAINERS:
        raise ValueError(
            f"Unsupported APV container {container!r}. "
            f"Choose one of: {', '.join(APV_CONTAINERS)}"
        )
    if not check_apv_available():
        raise RuntimeError(INSTALL_HINT)

    spec = APV_PRESETS[preset]
    qp = spec["qp"] if qp_override is None else str(int(qp_override))
    if not 0 <= int(qp) <= 51:
        raise ValueError("APV qp must be between 0 and 51")

    # output_path() keeps the source extension and treats its third argument
    # as an output directory, so swap the container explicitly.
    default_out = output_path(input_path, f"apv_{preset}")
    out = output or (os.path.splitext(default_out)[0] + container)
    if on_progress:
        on_progress(5)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-c:v", APV_ENCODER,
        "-qp", str(qp),
        "-preset", spec["preset"],
        "-pix_fmt", pix_fmt,
        # APV is all-intra; copy audio through untouched so a mezzanine
        # round-trip does not re-encode sound on every generation.
        "-c:a", "copy",
        out,
    ]
    run_ffmpeg(cmd, timeout=14400)
    if on_progress:
        on_progress(100)

    if not os.path.isfile(out):
        raise RuntimeError(f"APV encode produced no output at {out!r}")

    return ApvResult(
        output=out,
        preset=preset,
        qp=int(qp),
        container=container,
        pix_fmt=pix_fmt,
        size_bytes=os.path.getsize(out),
        notes=[
            "APV is an all-intra mezzanine codec (IETF RFC 9924); files are "
            "intentionally large relative to delivery codecs.",
        ],
    )
