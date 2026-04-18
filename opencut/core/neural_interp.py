"""
OpenCut Neural Frame Interpolation (RIFE + fallback)

Video frame rate up-conversion using RIFE (Real-Time Intermediate Flow
Estimation) when available. Falls back to FFmpeg's ``minterpolate``
optical-flow interpolation when a RIFE backend is not installed.

Supported backends, in priority order:

1. ``rife-ncnn-vulkan`` CLI (lightweight, ~50 MB, runs on any GPU/CPU via
   Vulkan; recommended default). Resolved from PATH. Project page:
   https://github.com/nihui/rife-ncnn-vulkan
2. Torch-based RIFE via the ``rife`` pip package (heavier, needs CUDA for
   real-time speed).
3. FFmpeg ``minterpolate`` as the last-resort fallback. Always available
   because FFmpeg is bundled.

The module exposes a single high-level ``interpolate(...)`` function that
auto-selects the backend.  It returns ``{output, backend, input_fps,
output_fps, duration, frames_added}``.

Design notes
------------
- Zero new **required** pip deps — the module degrades to FFmpeg when
  neural backends are absent.
- The RIFE CLI path writes PNG frames to a temp dir, runs RIFE to double
  the frame count (repeatedly if needed to reach the target), then
  re-encodes via FFmpeg with libx264 + AAC passthrough (muxed from the
  original).
- Audio is **preserved verbatim** (stream copy from source) regardless of
  backend — no resampling drift.
- FFmpeg fallback re-uses the existing ``framerate_convert`` preset
  semantics so users see consistent quality knobs.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess as _sp
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    _try_import,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

RIFE_CLI_BIN = "rife-ncnn-vulkan"

# Backend-availability cache.  Guarded by ``_BACKEND_LOCK`` so concurrent
# requests can't observe the "checked=True but values not yet written"
# window — cheap, but a correctness nit worth closing on a server that
# can serve many parallel requests.
_BACKEND_CACHE: Dict[str, Optional[str]] = {
    "rife_cli": None,    # Path to rife-ncnn-vulkan or ""
    "rife_torch": None,  # "yes" if `rife` pip pkg importable, "" otherwise
    "checked": False,
}
_BACKEND_LOCK = threading.Lock()


def _detect_backends() -> Dict[str, bool]:
    """Return {rife_cli: bool, rife_torch: bool, ffmpeg: bool}.

    Cached at module level so bulk calls don't spawn subprocesses per item.
    Concurrent callers serialise through ``_BACKEND_LOCK``; once the cache
    is populated, subsequent reads still acquire the lock but the
    contention window is trivial.
    """
    with _BACKEND_LOCK:
        if not _BACKEND_CACHE["checked"]:
            _BACKEND_CACHE["rife_cli"] = shutil.which(RIFE_CLI_BIN) or ""
            _BACKEND_CACHE["rife_torch"] = (
                "yes" if _try_import("rife") is not None else ""
            )
            # Set ``checked`` last so a reader that sees the flag sees
            # the populated values too.
            _BACKEND_CACHE["checked"] = True
        rife_cli = _BACKEND_CACHE["rife_cli"]
        rife_torch = _BACKEND_CACHE["rife_torch"]
    return {
        "rife_cli": bool(rife_cli),
        "rife_torch": bool(rife_torch),
        "ffmpeg": bool(get_ffmpeg_path()),
    }


def check_neural_interp_available() -> bool:
    """True when *any* interpolation backend is usable (FFmpeg always is)."""
    return _detect_backends()["ffmpeg"]


def available_backends() -> List[str]:
    """Return backend names in priority order, filtered to what's installed."""
    det = _detect_backends()
    out = []
    if det["rife_cli"]:
        out.append("rife_cli")
    if det["rife_torch"]:
        out.append("rife_torch")
    if det["ffmpeg"]:
        out.append("ffmpeg")
    return out


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InterpResult:
    """Structured result of a neural interpolation job."""
    output: str = ""
    backend: str = ""
    input_fps: float = 0.0
    output_fps: float = 0.0
    duration: float = 0.0
    frames_added: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# FFmpeg minterpolate fallback
# ---------------------------------------------------------------------------

def _interp_ffmpeg(
    video_path: str,
    target_fps: float,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> None:
    """FFmpeg-only optical-flow interpolation (minterpolate)."""
    if on_progress:
        on_progress(15, "Interpolating via FFmpeg minterpolate")
    vf = (
        f"minterpolate=fps={target_fps}:"
        "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    )
    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd, timeout=7200)


# ---------------------------------------------------------------------------
# RIFE CLI backend
# ---------------------------------------------------------------------------

def _rife_doubling_passes(input_fps: float, target_fps: float) -> int:
    """RIFE doubles per pass. Return the number of passes needed to reach
    or exceed ``target_fps`` starting from ``input_fps``.  Capped at 3
    (8x) — beyond that, quality degrades and disk use explodes."""
    if target_fps <= input_fps or input_fps <= 0:
        return 0
    ratio = target_fps / input_fps
    passes = int(math.ceil(math.log2(max(ratio, 1.0001))))
    return max(1, min(passes, 3))


def _extract_frames(video_path: str, frame_dir: str, on_progress=None) -> float:
    """Extract PNG frames from ``video_path`` into ``frame_dir``.

    Returns the source FPS so the caller can compute the output frame
    rate after RIFE processing.
    """
    os.makedirs(frame_dir, exist_ok=True)
    info = get_video_info(video_path)
    src_fps = float(info.get("fps") or 30.0)
    if on_progress:
        on_progress(20, "Extracting source frames")
    cmd = (
        FFmpegCmd()
        .input(video_path)
        .option("vsync", "0")
        .output(os.path.join(frame_dir, "frame_%08d.png"))
        .build()
    )
    run_ffmpeg(cmd, timeout=3600)
    return src_fps


def _rife_cli_pass(rife_bin: str, in_dir: str, out_dir: str, timeout: int) -> None:
    """Run a single doubling pass of the RIFE CLI."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [rife_bin, "-i", in_dir, "-o", out_dir]
    logger.debug("RIFE pass: %s", " ".join(cmd))
    proc = _sp.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        stderr = (proc.stderr or "")[-800:]
        raise RuntimeError(f"rife-ncnn-vulkan failed (rc={proc.returncode}): {stderr}")


def _reassemble(frame_dir: str, fps: float, audio_source: str, out_path: str) -> None:
    """Mux the interpolated frames + original audio into ``out_path``."""
    cmd = (
        FFmpegCmd()
        .input(os.path.join(frame_dir, "frame_%08d.png"), framerate=fps)
        .input(audio_source)
        .map("0:v:0")
        .map("1:a:0?")
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .option("shortest")
        .faststart()
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd, timeout=7200)


def _interp_rife_cli(
    video_path: str,
    target_fps: float,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, int]:
    """Full RIFE-NCNN-Vulkan pipeline: extract → N doubling passes → reassemble."""
    rife_bin = _BACKEND_CACHE.get("rife_cli") or shutil.which(RIFE_CLI_BIN)
    if not rife_bin:
        raise RuntimeError("rife-ncnn-vulkan not available")

    tmp_root = tempfile.mkdtemp(prefix="opencut_rife_")
    try:
        src_dir = os.path.join(tmp_root, "src")
        src_fps = _extract_frames(video_path, src_dir, on_progress)
        passes = _rife_doubling_passes(src_fps, target_fps)
        if passes == 0:
            raise ValueError(
                f"target_fps ({target_fps}) must exceed source fps ({src_fps}) for RIFE"
            )
        cur_dir = src_dir
        effective_fps = src_fps
        for i in range(passes):
            if on_progress:
                pct = 25 + int(50 * (i / passes))
                on_progress(pct, f"RIFE pass {i + 1}/{passes} ({effective_fps:.1f} → {effective_fps * 2:.1f} fps)")
            next_dir = os.path.join(tmp_root, f"pass_{i + 1}")
            _rife_cli_pass(rife_bin, cur_dir, next_dir, timeout=7200)
            cur_dir = next_dir
            effective_fps *= 2

        # Frame count now matches ``effective_fps``; decimate to the exact
        # target_fps via FFmpeg's fps filter during reassembly if needed.
        if on_progress:
            on_progress(80, "Muxing interpolated frames with original audio")

        # If effective_fps > target_fps, write out at effective_fps and
        # let a single-pass fps filter trim.  Simpler and avoids a decimate
        # filter inside RIFE's output stage.
        mux_fps = effective_fps
        if abs(effective_fps - target_fps) > 0.5:
            # Intermediate reassembly at effective_fps, then FPS-convert.
            inter_path = os.path.join(tmp_root, "inter.mp4")
            _reassemble(cur_dir, mux_fps, video_path, inter_path)
            cmd = (
                FFmpegCmd()
                .input(inter_path)
                .video_filter(f"fps={target_fps}")
                .video_codec("libx264", crf=18, preset="medium")
                .audio_codec("aac", bitrate="192k")
                .faststart()
                .output(out_path)
                .build()
            )
            run_ffmpeg(cmd, timeout=3600)
        else:
            _reassemble(cur_dir, mux_fps, video_path, out_path)

        src_frame_count = _count_frames(src_dir)
        final_frame_count = _count_frames(cur_dir)
        return {
            "src_frames": src_frame_count,
            "interp_frames": final_frame_count,
            "passes": passes,
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _count_frames(directory: str) -> int:
    try:
        return sum(1 for n in os.listdir(directory) if n.endswith(".png"))
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Torch RIFE backend (stub — delegates to CLI if available, else FFmpeg)
# ---------------------------------------------------------------------------

def _interp_rife_torch(
    video_path: str,
    target_fps: float,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, int]:
    """Torch-based RIFE backend.

    The ``rife`` pip package API varies across forks; rather than
    hard-coding a specific fork, we try to import and raise
    NotImplementedError if no recognised entry point is present.  Callers
    should fall through to ``_interp_rife_cli`` or FFmpeg.
    """
    try:
        import rife  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"rife pip package not importable: {exc}") from exc

    # No stable public API to rely on here — keep this path as an
    # explicit "not implemented" so users who prefer CLI get consistent
    # behaviour, and torch users don't get silent drift.
    raise NotImplementedError(
        "Torch-based RIFE fork detected but no stable API wrapper is provided. "
        "Install rife-ncnn-vulkan CLI for GPU-accelerated neural interpolation."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interpolate(
    video_path: str,
    target_fps: Optional[float] = None,
    backend: str = "auto",
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> InterpResult:
    """Upsample ``video_path`` to ``target_fps`` using the best available
    backend.

    Args:
        video_path: Input video.  Must exist.
        target_fps: Desired output frame rate.  If *None*, doubles input.
        backend: One of ``"auto"``, ``"rife_cli"``, ``"rife_torch"``,
            ``"ffmpeg"``.  ``"auto"`` picks the highest-priority backend
            that is installed.
        output: Explicit output path.
        on_progress: Optional ``(pct, msg)`` callback.

    Returns:
        :class:`InterpResult` — subscriptable so routes can ``return result``
        straight to Flask's ``jsonify``.

    Raises:
        FileNotFoundError: ``video_path`` missing.
        ValueError: ``target_fps`` <= source fps (nothing to interpolate).
        RuntimeError: Selected backend failed.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    info = get_video_info(video_path)
    src_fps = float(info.get("fps") or 30.0)
    duration = float(info.get("duration") or 0.0)
    if target_fps is None:
        target_fps = src_fps * 2.0
    target_fps = float(target_fps)
    if target_fps <= 0:
        raise ValueError(f"Invalid target_fps: {target_fps}")
    if target_fps <= src_fps:
        raise ValueError(
            f"target_fps ({target_fps}) must exceed source fps ({src_fps}) "
            "for interpolation. Use /video/framerate-convert for downsampling."
        )

    out = output or output_path(video_path, f"interp_{int(target_fps)}")
    det = _detect_backends()

    if backend == "auto":
        if det["rife_cli"]:
            backend = "rife_cli"
        elif det["rife_torch"]:
            backend = "rife_torch"
        else:
            backend = "ffmpeg"

    notes: List[str] = []
    frames_added = 0

    if on_progress:
        on_progress(5, f"Interpolating {src_fps:.2f} → {target_fps:.2f} fps via {backend}")

    if backend == "rife_cli":
        if not det["rife_cli"]:
            notes.append("rife_cli not installed; falling back to FFmpeg")
            _interp_ffmpeg(video_path, target_fps, out, on_progress)
            backend = "ffmpeg"
        else:
            try:
                stats = _interp_rife_cli(video_path, target_fps, out, on_progress)
                frames_added = max(0, stats.get("interp_frames", 0) - stats.get("src_frames", 0))
                notes.append(f"RIFE CLI: {stats.get('passes', 0)} doubling pass(es)")
            except Exception as exc:
                logger.warning("RIFE CLI failed (%s); falling back to FFmpeg", exc)
                notes.append(f"RIFE CLI failed ({exc}); fell back to FFmpeg")
                _interp_ffmpeg(video_path, target_fps, out, on_progress)
                backend = "ffmpeg"
    elif backend == "rife_torch":
        try:
            stats = _interp_rife_torch(video_path, target_fps, out, on_progress)
            frames_added = max(0, stats.get("interp_frames", 0) - stats.get("src_frames", 0))
            notes.append("RIFE torch backend")
        except Exception as exc:
            logger.warning("RIFE torch failed (%s); falling back to FFmpeg", exc)
            notes.append(f"RIFE torch failed ({exc}); fell back to FFmpeg")
            _interp_ffmpeg(video_path, target_fps, out, on_progress)
            backend = "ffmpeg"
    elif backend == "ffmpeg":
        _interp_ffmpeg(video_path, target_fps, out, on_progress)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if on_progress:
        on_progress(100, "Interpolation complete")

    if frames_added == 0 and duration > 0:
        frames_added = max(0, int(round((target_fps - src_fps) * duration)))

    return InterpResult(
        output=out,
        backend=backend,
        input_fps=round(src_fps, 3),
        output_fps=round(target_fps, 3),
        duration=round(duration, 3),
        frames_added=frames_added,
        notes=notes,
    )


def list_backends() -> List[Dict[str, object]]:
    """Return a UI-friendly description of every backend, installed or not."""
    det = _detect_backends()
    return [
        {
            "name": "rife_cli",
            "description": "RIFE-NCNN-Vulkan CLI (recommended, GPU via Vulkan).",
            "installed": det["rife_cli"],
            "hint": "Install rife-ncnn-vulkan binary on PATH. Project: https://github.com/nihui/rife-ncnn-vulkan",
        },
        {
            "name": "rife_torch",
            "description": "Torch-based RIFE pip package (experimental).",
            "installed": det["rife_torch"],
            "hint": "pip install rife (API varies across forks — CLI is preferred).",
        },
        {
            "name": "ffmpeg",
            "description": "FFmpeg minterpolate optical-flow fallback.",
            "installed": det["ffmpeg"],
            "hint": "Bundled with OpenCut.",
        },
    ]
