"""
VVC / H.266 export via Fraunhofer libvvenc.

Wraps FFmpeg's ``libvvenc`` encoder (Fraunhofer HHI,
https://github.com/fraunhoferhhi/vvenc, BSD-3) for VVC / H.266 output.
VVC delivers ~30 % smaller files than HEVC at equal VMAF, at the cost
of significantly slower encode — so it's targeted at archival /
delivery, not real-time.

Availability is probed at import time: if the linked FFmpeg wasn't
built with ``--enable-libvvenc``, the encoder is absent and callers
see a clear ``MISSING_DEPENDENCY`` error.  Three preset packs map to
battle-tested trade-offs:

- ``faster``  — preset 3 / QP 35. Roughly H.265-slow speed, ~15 % smaller.
- ``balanced`` — preset 2 / QP 32. Recommended default.
- ``archive`` — preset 1 / QP 27. Visually transparent archival masters.

Audio is AAC-copied passthrough when already AAC, otherwise re-encoded
to 192 kbps AAC to stay in MP4-compatible territory.  Container is
``.mp4`` (ISO BMFF) — ``.265`` raw-stream output is offered as an
opt-in for HLS packagers.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Preset packs
# ---------------------------------------------------------------------------

VVC_PRESETS: Dict[str, Dict[str, str]] = {
    "faster": {
        "description": "Faster preset — HEVC-slow speed, ~15% smaller at equal VMAF.",
        "preset": "3",
        "qp": "35",
    },
    "balanced": {
        "description": "Balanced preset — the recommended default.",
        "preset": "2",
        "qp": "32",
    },
    "archive": {
        "description": "Archive preset — visually transparent, slow encode.",
        "preset": "1",
        "qp": "27",
    },
}

# Allowed output container extensions
_VALID_CONTAINERS = frozenset({".mp4", ".mkv", ".266", ".vvc"})


@dataclass
class VvcResult:
    """Structured return from a VVC encode."""
    output: str = ""
    preset: str = "balanced"
    qp: int = 32
    duration: float = 0.0
    container: str = ".mp4"
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

_AVAILABILITY_CACHE: Dict[str, Optional[bool]] = {"libvvenc": None}


def check_vvc_available() -> bool:
    """Cache-lookup probe of FFmpeg for the ``libvvenc`` encoder."""
    if _AVAILABILITY_CACHE["libvvenc"] is not None:
        return bool(_AVAILABILITY_CACHE["libvvenc"])
    ff = get_ffmpeg_path()
    if not ff:
        _AVAILABILITY_CACHE["libvvenc"] = False
        return False
    try:
        proc = _sp.run(
            [ff, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        present = bool(re.search(r"^\s*\S*\s+libvvenc\b",
                                  (proc.stdout or ""), flags=re.MULTILINE))
        _AVAILABILITY_CACHE["libvvenc"] = present
        return present
    except Exception:  # noqa: BLE001
        _AVAILABILITY_CACHE["libvvenc"] = False
        return False


def list_presets() -> List[Dict[str, str]]:
    return [{"name": k, **v} for k, v in VVC_PRESETS.items()]


# ---------------------------------------------------------------------------
# Encode driver
# ---------------------------------------------------------------------------

def encode_vvc(
    input_path: str,
    preset: str = "balanced",
    output: Optional[str] = None,
    qp_override: Optional[int] = None,
    container: str = ".mp4",
    on_progress: Optional[Callable] = None,
) -> VvcResult:
    """Encode ``input_path`` to VVC / H.266.

    Args:
        input_path: Any video FFmpeg can decode.
        preset: One of :data:`VVC_PRESETS` keys.
        output: Output path. Defaults to ``<input>_vvc.<container>``.
        qp_override: Override the preset's quantiser (0..63). Lower =
            higher quality.
        container: ``.mp4`` (default) / ``.mkv`` / ``.266`` / ``.vvc``.
            ``.266`` / ``.vvc`` emit a raw VVC elementary stream without
            audio — suitable for HLS packagers.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`VvcResult`.

    Raises:
        RuntimeError: FFmpeg missing libvvenc, or encode failed.
        ValueError: unknown preset / invalid container.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if preset not in VVC_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Valid: {list(VVC_PRESETS.keys())}"
        )
    container = (container or ".mp4").lower()
    if container not in _VALID_CONTAINERS:
        raise ValueError(
            f"Unsupported container '{container}'. Valid: "
            f"{sorted(_VALID_CONTAINERS)}"
        )
    if not check_vvc_available():
        raise RuntimeError(
            "FFmpeg was not built with --enable-libvvenc. Install a build "
            "with libvvenc linked (e.g. Fraunhofer HHI upstream) or use "
            "/video/encode/svtav1-psy for AV1 instead."
        )

    cfg = VVC_PRESETS[preset]
    qp = int(qp_override) if qp_override is not None else int(cfg["qp"])
    if not (0 <= qp <= 63):
        raise ValueError("qp must be in [0, 63]")

    base_suffix = f"vvc_{preset}"
    default_out = output_path(input_path, base_suffix)
    # output_path() keeps the source extension; swap to requested container
    default_base, _orig_ext = os.path.splitext(default_out)
    default_out = default_base + container

    out = output or default_out

    if on_progress:
        on_progress(5, f"Encoding VVC ({preset}, QP {qp}) → {container}")

    raw_stream = container in (".266", ".vvc")

    cmd_builder = (
        FFmpegCmd()
        .input(input_path)
        .video_codec("libvvenc")
        .option("preset", cfg["preset"])
        .option("qp", qp)
        .option("pix_fmt", "yuv420p10le")
    )

    if raw_stream:
        # Raw VVC elementary stream — no audio, container-free (HLS pkg'rs).
        cmd = (
            cmd_builder
            .option("an")
            .format("h266")
            .output(out)
            .build()
        )
    else:
        # MP4 / MKV with audio
        cmd = (
            cmd_builder
            .audio_codec("aac", bitrate="192k")
            .faststart()
            .output(out)
            .build()
        )

    logger.debug("VVC cmd: %s", " ".join(cmd))

    if on_progress:
        on_progress(15, "Handing off to FFmpeg (libvvenc)…")

    # VVC is slow — give the encoder 4 h before we give up.
    run_ffmpeg(cmd, timeout=14400)

    duration = 0.0
    try:
        from opencut.helpers import get_video_info
        info = get_video_info(out)
        duration = float(info.get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        pass

    if on_progress:
        on_progress(100, "VVC encode complete")

    return VvcResult(
        output=out,
        preset=preset,
        qp=qp,
        duration=round(duration, 3),
        container=container,
        notes=[cfg.get("description", ""), f"libvvenc preset={cfg['preset']}"],
    )
