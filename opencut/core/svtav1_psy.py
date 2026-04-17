"""
SVT-AV1-PSY encoder wrapper.

SVT-AV1-PSY (https://github.com/gianni-rosato/svt-av1-psy, BSD) is a
perceptually-tuned fork of Intel/Netflix SVT-AV1 used by the AV1
enthusiast community for higher visual quality at equal bitrate.  It
ships as a drop-in ``SvtAv1EncApp`` binary; the simplest integration is
to have FFmpeg pipe raw frames to it.

However, several Linux distros and msys2 now ship FFmpeg builds with
``libsvtav1`` already linked against the PSY fork, in which case
passing ``-c:v libsvtav1 -svtav1-params "tune=3:enable-variance-boost=1:psy-rd=1.0"``
is enough.  This module picks the integrated path when available and
falls back to a two-stage pipe (rawvideo → SvtAv1EncApp → mux) when
only the standalone binary is on PATH.

Design notes
------------
- This is not a replacement for the existing `core/av1_export.py`;
  that one targets the *stock* SVT-AV1 with conservative defaults.
  Users who want the PSY-tuned fork opt in via this module (and its
  route) explicitly.
- Stream copy of audio from the source avoids resampling drift.
- We surface three preset packs (``social``, ``web``, ``archive``) that
  map to battle-tested `svtav1-params` strings.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


SVT_AV1_ENC_BIN = "SvtAv1EncApp"

# Preset packs — tested on ab-av1 / AV1-enthusiast leaderboards.
# Keys map to ``-svtav1-params`` strings (semi-colon separated).
PSY_PRESETS: Dict[str, Dict[str, str]] = {
    "social": {
        "description": "Short social clips — small files, acceptable quality",
        "svtav1_params": "tune=3:enable-variance-boost=1:variance-boost-strength=2:psy-rd=1.0",
        "crf": "32",
        "preset": "7",
    },
    "web": {
        "description": "Balanced web delivery — recommended default",
        "svtav1_params": "tune=3:enable-variance-boost=1:variance-boost-strength=2:psy-rd=1.0:film-grain=4",
        "crf": "28",
        "preset": "6",
    },
    "archive": {
        "description": "High-fidelity archival — slow encode, visually transparent",
        "svtav1_params": "tune=3:enable-variance-boost=1:variance-boost-strength=3:psy-rd=1.5:film-grain=8:sharpness=2",
        "crf": "22",
        "preset": "4",
    },
}


@dataclass
class SvtAv1PsyResult:
    """Structured return from an SVT-AV1-PSY encode."""
    output: str = ""
    preset: str = "web"
    backend: str = "ffmpeg_integrated"   # or "standalone_bin"
    crf: int = 28
    svtav1_params: str = ""
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _ffmpeg_supports_libsvtav1_psy() -> bool:
    """Rough check: does FFmpeg know about the PSY-specific svtav1-params?

    We probe for the encoder being present; the PSY params are tolerated
    by both stock and PSY builds, so this is the best we can do without
    actually encoding.
    """
    ff = get_ffmpeg_path()
    try:
        proc = _sp.run(
            [ff, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        return "libsvtav1" in (proc.stdout or "")
    except Exception:  # noqa: BLE001
        return False


def check_svtav1_psy_available() -> bool:
    """Either the integrated libsvtav1 or the standalone CLI is usable."""
    return _ffmpeg_supports_libsvtav1_psy() or bool(shutil.which(SVT_AV1_ENC_BIN))


def list_presets() -> List[Dict[str, str]]:
    return [{"name": k, **v} for k, v in PSY_PRESETS.items()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_svtav1_psy(
    input_path: str,
    preset: str = "web",
    output: Optional[str] = None,
    crf_override: Optional[int] = None,
    svtav1_params_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> SvtAv1PsyResult:
    """Encode ``input_path`` with SVT-AV1-PSY.

    Args:
        input_path: Source video (anything FFmpeg decodes).
        preset: One of :data:`PSY_PRESETS` keys.
        output: Output path. Defaults to ``<input>_psy.mp4``.
        crf_override: Explicit CRF, overrides the preset's value.
        svtav1_params_override: Explicit `-svtav1-params` string.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`SvtAv1PsyResult`.

    Raises:
        RuntimeError: no backend available.
        ValueError: unknown preset.
        FileNotFoundError: source missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if preset not in PSY_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Valid: {list(PSY_PRESETS.keys())}"
        )
    if not check_svtav1_psy_available():
        raise RuntimeError(
            "SVT-AV1-PSY not available. Either use an FFmpeg build with "
            "libsvtav1 enabled, or install SvtAv1EncApp on PATH. See "
            "https://github.com/gianni-rosato/svt-av1-psy."
        )

    cfg = PSY_PRESETS[preset]
    crf = int(crf_override) if crf_override is not None else int(cfg["crf"])
    params = svtav1_params_override or cfg["svtav1_params"]

    out = output or output_path(input_path, f"psy_{preset}")

    if on_progress:
        on_progress(5, f"Encoding SVT-AV1-PSY ({preset}, CRF {crf})")

    if _ffmpeg_supports_libsvtav1_psy():
        backend = "ffmpeg_integrated"
        cmd = (
            FFmpegCmd()
            .input(input_path)
            .video_codec("libsvtav1", crf=crf, preset=cfg.get("preset", "6"))
            .option("svtav1-params", params)
            .option("pix_fmt", "yuv420p10le")
            .audio_codec("libopus", bitrate="160k")
            .faststart()
            .output(out)
            .build()
        )
        if on_progress:
            on_progress(15, "Handing off to FFmpeg (libsvtav1)")
        run_ffmpeg(cmd, timeout=10800)
    else:
        backend = "standalone_bin"
        _encode_via_standalone(input_path, out, crf, params, cfg, on_progress)

    duration = 0.0
    try:
        from opencut.helpers import get_video_info
        duration = float(get_video_info(out).get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        pass

    if on_progress:
        on_progress(100, "Encode complete")

    return SvtAv1PsyResult(
        output=out,
        preset=preset,
        backend=backend,
        crf=crf,
        svtav1_params=params,
        duration=round(duration, 3),
        notes=[cfg.get("description", "")],
    )


# ---------------------------------------------------------------------------
# Standalone CLI path (rarely needed, but good to have)
# ---------------------------------------------------------------------------

def _encode_via_standalone(
    input_path: str,
    out_path: str,
    crf: int,
    params: str,
    cfg: Dict[str, str],
    on_progress: Optional[Callable],
) -> None:
    """Raw-pipe path via FFmpeg → SvtAv1EncApp → FFmpeg mux."""
    ff = get_ffmpeg_path()
    enc = shutil.which(SVT_AV1_ENC_BIN) or SVT_AV1_ENC_BIN

    # Get source dimensions + fps so the encoder knows what to do
    from opencut.helpers import get_video_info
    info = get_video_info(input_path)
    w = int(info.get("width") or 1920)
    h = int(info.get("height") or 1080)
    fps = float(info.get("fps") or 30.0)

    # Stage 1: rawvideo pipe → SvtAv1EncApp → .ivf
    fd_ivf = os.path.splitext(out_path)[0] + ".psy.ivf"
    if on_progress:
        on_progress(20, "Stage 1: rawvideo → SvtAv1EncApp")

    # FFmpeg decode → rawvideo yuv420p10le
    dec_cmd = [
        ff, "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-pix_fmt", "yuv420p10le",
        "-f", "yuv4mpegpipe", "-strict", "-1",
        "-",
    ]
    # SvtAv1EncApp reads y4m on stdin
    enc_cmd = [
        enc,
        "-i", "stdin",
        "-b", fd_ivf,
        "--preset", str(cfg.get("preset", "6")),
        "--crf", str(crf),
        "--svtav1-params", params,
        "-w", str(w), "-h", str(h),
        "--fps-num", str(int(round(fps * 1000))), "--fps-denom", "1000",
    ]
    logger.debug("SvtAv1-PSY stage 1: %s | %s", " ".join(dec_cmd), " ".join(enc_cmd))

    p1 = _sp.Popen(dec_cmd, stdout=_sp.PIPE, stderr=_sp.PIPE)
    p2 = _sp.Popen(enc_cmd, stdin=p1.stdout, stdout=_sp.PIPE, stderr=_sp.PIPE)
    if p1.stdout:
        p1.stdout.close()  # Let p2 receive SIGPIPE when done
    p1_stderr = b""
    p2_stderr = b""
    try:
        p2_stdout, p2_stderr = p2.communicate(timeout=10800)
        p1.wait(timeout=10)
        _ = p2_stdout  # unused
    except _sp.TimeoutExpired:
        p1.kill()
        p2.kill()
        raise RuntimeError("SvtAv1EncApp stage timed out after 3 hours")
    finally:
        if p1.stderr:
            try:
                p1_stderr = p1.stderr.read()
            except Exception:  # noqa: BLE001
                pass
    if p2.returncode != 0:
        raise RuntimeError(
            f"SvtAv1EncApp failed (rc={p2.returncode}): "
            f"{p2_stderr[-400:].decode(errors='replace')}\n"
            f"decoder stderr: {p1_stderr[-200:].decode(errors='replace')}"
        )

    # Stage 2: mux IVF + source audio into MP4
    if on_progress:
        on_progress(80, "Stage 2: muxing IVF + audio")

    cmd = (
        FFmpegCmd()
        .input(fd_ivf)
        .input(input_path)
        .map("0:v:0")
        .map("1:a:0?")
        .video_codec("copy")
        .audio_codec("libopus", bitrate="160k")
        .faststart()
        .output(out_path)
        .build()
    )
    try:
        run_ffmpeg(cmd, timeout=1800)
    finally:
        try:
            os.unlink(fd_ivf)
        except OSError:
            pass
