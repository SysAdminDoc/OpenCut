"""
OpenCut Lottie Import v1.28.0

Render .json/.lottie Lottie animations to video with alpha.
Requires: pip install lottie
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import _try_import, get_ffmpeg_path

logger = logging.getLogger("opencut")
INSTALL_HINT = "pip install lottie"


def check_lottie_available() -> bool:
    return _try_import("lottie") is not None


@dataclass
class LottieResult:
    output: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    frames: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("output", "width", "height", "fps", "duration", "frames", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def info(lottie_path: str) -> Dict:
    with open(lottie_path, "rb") as f:
        header = f.read(4)
    if header[:2] == b"PK":
        with zipfile.ZipFile(lottie_path) as zf:
            names = zf.namelist()
            json_name = next((n for n in names if n.endswith(".json")), names[0])
            data = json.loads(zf.read(json_name))
    else:
        with open(lottie_path, encoding="utf-8") as f:
            data = json.load(f)
    w = int(data.get("w", 1920))
    h = int(data.get("h", 1080))
    fps = float(data.get("fr", 30))
    ip = float(data.get("ip", 0))
    op = float(data.get("op", 90))
    frames = int(op - ip)
    duration = frames / fps if fps > 0 else 0.0
    return {"width": w, "height": h, "fps": fps, "duration": duration,
            "frames": frames, "version": str(data.get("v", ""))}


def render(
    lottie_path: str,
    output: Optional[str] = None,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    bg_color: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> LottieResult:
    if not check_lottie_available():
        raise RuntimeError(f"lottie is not installed. Install with:\n    {INSTALL_HINT}")

    import lottie  # type: ignore
    from lottie import exporters  # type: ignore

    meta = info(lottie_path)
    nat_fps = meta["fps"]
    frames_meta = meta["frames"]
    duration = meta["duration"]

    if output is None:
        base = os.path.splitext(lottie_path)[0]
        output = f"{base}.webm"

    if on_progress:
        on_progress(5, "Parsing Lottie animation")

    anim = lottie.objects.Animation.load(lottie_path)
    frame_dir = tempfile.mkdtemp(prefix="lottie_frames_")
    total_frames = int(duration * fps)

    try:
        for fi in range(total_frames):
            t = fi / fps
            frame_num = int(t * nat_fps)
            png_path = os.path.join(frame_dir, f"frame_{fi:06d}.png")
            if hasattr(exporters, "cairo"):
                exporters.cairo.export_frame(anim, png_path, frame_num, w=width, h=height)
            else:
                # Blank-frame fallback is not useful; require a real rendering backend.
                raise RuntimeError(
                    "Cairo rendering backend not available. "
                    "Install with: pip install lottie[cairo] or pip install cairocffi"
                )
            if on_progress and fi % max(1, total_frames // 20) == 0:
                on_progress(int(5 + fi / total_frames * 80), f"Rendering frame {fi}/{total_frames}")

        if on_progress:
            on_progress(85, "Encoding video")

        frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
        ffmpeg = get_ffmpeg_path()
        if bg_color:
            # Composite RGBA frames over a solid background colour, then encode H.264
            fc = (
                f"color=c={bg_color}:s={width}x{height}:r={fps}[bg];"
                f"[0:v]format=argb[fg];"
                f"[bg][fg]overlay[out]"
            )
            cmd = [
                ffmpeg, "-y", "-framerate", str(fps), "-i", frame_pattern,
                "-filter_complex", fc, "-map", "[out]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", output,
            ]
        else:
            cmd = [
                ffmpeg, "-y", "-framerate", str(fps), "-i", frame_pattern,
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30", output,
            ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode(errors="replace") if e.stderr else ""
            raise RuntimeError(f"FFmpeg encode failed (exit {e.returncode}): {stderr_msg[:500]}") from e
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    if on_progress:
        on_progress(100, "Done")

    return LottieResult(output=output, width=width, height=height, fps=fps,
                        duration=duration, frames=total_frames, notes=[])


__all__ = ["check_lottie_available", "INSTALL_HINT", "LottieResult", "render", "info"]
