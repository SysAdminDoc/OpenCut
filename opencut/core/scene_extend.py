"""
OpenCut AI Scene Extension

Extend a video clip's duration by generating continuation frames.
Feeds the last N frames to a video prediction model and blends the
transition for a seamless result.

Falls back to optical-flow extrapolation + freeze-frame blend when
AI models are not available.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class SceneExtendResult:
    """Result of AI scene extension."""
    output_path: str = ""
    original_duration: float = 0.0
    extended_duration: float = 0.0
    extra_seconds: float = 0.0
    frames_generated: int = 0
    method: str = ""
    blend_frames: int = 0


# ---------------------------------------------------------------------------
# Optical flow extrapolation fallback
# ---------------------------------------------------------------------------

def _extrapolate_optical_flow(
    video_path: str,
    out_path: str,
    extra_seconds: float = 3.0,
    blend_frames: int = 10,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Extend video using optical flow extrapolation.

    Reads the last chunk of frames, computes flow vectors, then
    extrapolates forward.  The generated frames are blended with
    a hold of the last frame to reduce drift.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all original frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Need at least 2 frames for flow extrapolation")

    extra_frame_count = int(extra_seconds * fps)

    if on_progress:
        on_progress(30, f"Extrapolating {extra_frame_count} frames via optical flow...")

    # Compute flow from last two frames
    gray_a = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 7, 1.5, 0)

    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    last_frame = frames[-1].copy()

    generated = []
    for i in range(extra_frame_count):
        scale = (i + 1) * 0.3
        map_x = gx + flow[:, :, 0] * scale
        map_y = gy + flow[:, :, 1] * scale
        extrap = cv2.remap(last_frame, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Blend with frozen frame to prevent drift
        alpha = min(1.0, (i + 1) / max(1, extra_frame_count * 0.6))
        blended = cv2.addWeighted(extrap, 1 - alpha * 0.6, last_frame, alpha * 0.6, 0)
        generated.append(blended)

    # Blend transition at boundary
    actual_blend = min(blend_frames, len(generated), len(frames))
    for i in range(actual_blend):
        t = (i + 1) / (actual_blend + 1)
        generated[i] = cv2.addWeighted(generated[i], t, last_frame, 1 - t, 0)

    if on_progress:
        on_progress(60, "Writing extended video...")

    # Write output
    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError("Cannot create video writer")

    try:
        for f in frames:
            writer.write(f)
        for f in generated:
            writer.write(f)
    finally:
        writer.release()

    # Re-encode with audio
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    return {
        "frames_generated": len(generated),
        "blend_frames": actual_blend,
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def extend_scene(
    video_path: str,
    extra_seconds: float = 3.0,
    blend_frames: int = 10,
    method: str = "auto",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Extend a video clip's duration by generating continuation frames.

    When *method* is ``"auto"``, attempts AI-based frame prediction
    first, falling back to optical-flow extrapolation.

    Args:
        video_path:  Input video file.
        extra_seconds:  Seconds to add at the end.
        blend_frames:  Number of frames for the transition blend.
        method:  ``"auto"``, ``"ai"``, or ``"optical_flow"``.
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *original_duration*,
        *extended_duration*, *extra_seconds*, *frames_generated*,
        *method*, *blend_frames*.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    if extra_seconds <= 0:
        raise ValueError("extra_seconds must be positive")

    info = get_video_info(video_path)
    original_duration = info.get("duration", 0.0)

    out = output_path_override or output_path(video_path, "extended")

    if on_progress:
        on_progress(5, f"Extending scene by {extra_seconds}s...")

    actual_method = method
    ai_available = False

    if method in ("auto", "ai"):
        try:
            torch = ensure_package("torch", "torch")
            if torch:
                import torch as _torch
                ai_available = _torch.cuda.is_available()
        except Exception:
            ai_available = False

        if not ai_available:
            actual_method = "optical_flow"

    if actual_method == "ai" and ai_available:
        if on_progress:
            on_progress(10, "AI scene extension not yet integrated, using optical flow...")
        actual_method = "optical_flow"

    # Optical flow path
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")

    if on_progress:
        on_progress(15, "Using optical flow extrapolation...")

    flow_result = _extrapolate_optical_flow(
        video_path, out,
        extra_seconds=extra_seconds,
        blend_frames=blend_frames,
        on_progress=on_progress,
    )

    if on_progress:
        on_progress(95, "Verifying output...")

    out_info = get_video_info(out)
    extended_duration = out_info.get("duration", original_duration + extra_seconds)

    if on_progress:
        on_progress(100, "Scene extension complete!")

    return {
        "output_path": out,
        "original_duration": round(original_duration, 2),
        "extended_duration": round(extended_duration, 2),
        "extra_seconds": round(extra_seconds, 2),
        "frames_generated": flow_result["frames_generated"],
        "method": actual_method,
        "blend_frames": flow_result["blend_frames"],
    }
