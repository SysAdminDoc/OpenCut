"""
OpenCut CineFocus Rack Focus (L3.5)

Depth-of-field bokeh using Depth Anything 3 or Depth Anything V2 depth:
keyframeable focal point, aperture shape, f-number slider, rack-focus
animation (focus-pull from background to foreground over N frames).

DaVinci Resolve 21 ships this as a premium AI feature. OpenCut ships free.

Licence: Apache-2.0 (Depth Anything 3 and V2)
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "pip install opencut-ppro[depth] for Depth Anything V2; "
    "optionally pip install depth-anything-3==0.1.1 for Depth Anything 3"
)

CINEFOCUS_PRESETS = {
    "shallow": {"aperture_f": 1.4, "description": "Very shallow DOF — strong background blur"},
    "portrait": {"aperture_f": 2.8, "description": "Portrait-style — subject sharp, bg soft"},
    "standard": {"aperture_f": 5.6, "description": "Standard — moderate background separation"},
    "deep": {"aperture_f": 11.0, "description": "Deep focus — most of frame in focus"},
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CineFocusResult:
    output: str = ""
    focal_z: float = 0.5
    aperture_f: float = 2.8
    duration: float = 0.0
    frames_processed: int = 0
    depth_backend: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "focal_z", "aperture_f", "duration",
                "frames_processed", "depth_backend", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_cinefocus_available() -> bool:
    """Return True when torch and at least one depth backend are available."""
    if _try_import("torch") is None:
        return False
    return _try_import("depth_anything_3") is not None or _try_import("transformers") is not None


# ---------------------------------------------------------------------------
# Depth backend helpers
# ---------------------------------------------------------------------------

def _get_depth_backend() -> str:
    """Detect which depth estimation backend is available."""
    if _try_import("depth_anything_3") is not None:
        return "depth-anything-3"
    try:
        import transformers

        if hasattr(transformers, "AutoModelForDepthEstimation"):
            return "depth-anything-v2"
    except Exception:
        pass
    return "unknown"


def _load_depth_backend():
    """Load DA3 when installed, otherwise use the Transformers DA2 adapter.

    DA3 has its own inference API; it is not an
    ``AutoModelForDepthEstimation`` architecture. Any DA3 import/model-load
    failure therefore falls back to the stable V2 path without breaking the
    CineFocus workflow.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from opencut.core.depth_anything_3 import (
            MODEL_ID,
            check_depth_anything_3_available,
        )

        if check_depth_anything_3_available():
            from depth_anything_3.api import DepthAnything3

            model = DepthAnything3.from_pretrained(MODEL_ID).to(device=device)
            if hasattr(model, "eval"):
                model.eval()
            return model, None, device, "depth-anything-3"
    except Exception as exc:
        logger.warning("Depth Anything 3 load failed; using V2 fallback: %s", exc)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, processor, device, "depth-anything-v2"


def _estimate_depth_frame(frame, model, processor, device):
    """Estimate depth for a single frame, returning a normalized depth map."""
    import numpy as np
    from PIL import Image

    if hasattr(frame, "shape"):  # numpy/cv2 array
        img = Image.fromarray(frame[..., ::-1])  # BGR->RGB
    else:
        img = frame

    if processor is None:
        prediction = model.inference([img])
        depth = np.asarray(prediction.depth).squeeze()
    else:
        import torch

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()

    # Normalize either backend to the 0-1 map consumed by the bokeh renderer.
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def _apply_bokeh(frame, depth_map, focal_z: float, aperture_f: float):
    """Apply depth-of-field blur based on depth map and focal plane."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    depth_resized = cv2.resize(depth_map, (w, h))

    # Distance from focal plane — closer to 0 = in focus
    distance = np.abs(depth_resized - focal_z)

    # Blur strength scales with distance from focal plane and aperture
    # Larger aperture (smaller f-number) = more blur
    max_blur = max(1, int(40.0 / max(aperture_f, 0.5)))
    blur_map = (distance * max_blur).astype(np.uint8)

    # Apply variable blur using multiple passes at different kernel sizes
    result = frame.copy()
    for blur_level in range(1, max_blur + 1, 2):
        if blur_level < 3:
            continue
        mask = (blur_map >= blur_level - 1) & (blur_map <= blur_level + 1)
        if not mask.any():
            continue
        blurred = cv2.GaussianBlur(frame, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = np.where(mask_3ch, blurred, result)

    return result


# ---------------------------------------------------------------------------
# Preview (single frame)
# ---------------------------------------------------------------------------

def preview(
    video_path: str,
    focal_z: float = 0.5,
    aperture_f: float = 2.8,
    frame: int = 0,
) -> Dict:
    """Generate a single-frame CineFocus preview.

    Returns dict with preview_path and depth info.
    """
    if not check_cinefocus_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")

    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame}")
    finally:
        cap.release()

    import torch

    model, processor, device, backend = _load_depth_backend()

    depth = _estimate_depth_frame(bgr, model, processor, device)
    result_frame = _apply_bokeh(bgr, depth, focal_z, aperture_f)

    fd, preview_path = tempfile.mkstemp(suffix=".jpg", prefix="opencut_cinefocus_")
    os.close(fd)
    cv2.imwrite(preview_path, result_frame)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "preview_path": preview_path,
        "focal_z": focal_z,
        "aperture_f": aperture_f,
        "depth_range": [float(depth.min()), float(depth.max())],
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Full render
# ---------------------------------------------------------------------------

def render(
    video_path: str,
    focal_z_start: float = 0.5,
    focal_z_end: float = 0.5,
    focal_frame_start: int = 0,
    focal_frame_end: int = 0,
    aperture_f: float = 2.8,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> CineFocusResult:
    """Render full CineFocus rack-focus video.

    Args:
        video_path: Source video path.
        focal_z_start: Depth focal plane at animation start (0=near, 1=far).
        focal_z_end: Depth focal plane at animation end.
        focal_frame_start: Frame where rack-focus animation begins.
        focal_frame_end: Frame where rack-focus animation ends (0=last frame).
        aperture_f: Aperture f-number (lower = more blur).
        output: Output path. Auto-generated if None.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        CineFocusResult with output path and metadata.
    """
    if not check_cinefocus_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")

    import cv2
    import torch

    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")

    focal_z_start = max(0.0, min(1.0, float(focal_z_start)))
    focal_z_end = max(0.0, min(1.0, float(focal_z_end)))
    aperture_f = max(0.5, min(22.0, float(aperture_f)))

    if on_progress:
        on_progress(5, "Loading depth model...")

    model, processor, device, backend = _load_depth_backend()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if focal_frame_end <= 0:
        focal_frame_end = total_frames - 1

    if not output:
        ext = os.path.splitext(video_path)[1] or ".mp4"
        fd, output = tempfile.mkstemp(suffix=ext, prefix="opencut_cinefocus_")
        os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    notes: List[str] = []
    notes.append(f"Backend: {backend}")
    notes.append(f"Aperture: f/{aperture_f}")

    if on_progress:
        on_progress(10, f"Processing {total_frames} frames...")

    start_time = time.monotonic()
    frame_idx = 0
    frames_processed = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            # Interpolate focal_z for rack-focus animation
            if focal_frame_end > focal_frame_start:
                t = max(0.0, min(1.0,
                    (frame_idx - focal_frame_start) / (focal_frame_end - focal_frame_start)
                ))
            else:
                t = 0.0
            current_focal = focal_z_start + (focal_z_end - focal_z_start) * t

            # Estimate depth every 3rd frame for speed, reuse last depth map
            if frame_idx % 3 == 0 or frame_idx == 0:
                depth = _estimate_depth_frame(bgr, model, processor, device)

            result_frame = _apply_bokeh(bgr, depth, current_focal, aperture_f)
            writer.write(result_frame)
            frames_processed += 1
            frame_idx += 1

            if on_progress and total_frames > 0 and frame_idx % 10 == 0:
                pct = 10 + int((frame_idx / total_frames) * 85)
                on_progress(min(95, pct), f"Frame {frame_idx}/{total_frames}")
    finally:
        cap.release()
        writer.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - start_time
    notes.append(f"Rendered in {elapsed:.1f}s ({frames_processed} frames)")

    # Mux audio from original
    try:
        from opencut.helpers import get_ffmpeg_path, run_ffmpeg
        fd2, muxed = tempfile.mkstemp(suffix=".mp4", prefix="opencut_cf_mux_")
        os.close(fd2)
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", output, "-i", video_path,
            "-c:v", "copy", "-map", "0:v", "-map", "1:a",
            "-shortest", muxed,
        ])
        os.replace(muxed, output)
        notes.append("Audio muxed from source")
    except Exception as exc:
        logger.warning("Audio mux failed (output has no audio): %s", exc)
        notes.append("No audio (mux failed)")

    duration = frames_processed / fps if fps > 0 else 0.0

    if on_progress:
        on_progress(100, "Done")

    return CineFocusResult(
        output=output,
        focal_z=focal_z_end,
        aperture_f=aperture_f,
        duration=round(duration, 2),
        frames_processed=frames_processed,
        depth_backend=backend,
        notes=notes,
    )


__all__ = [
    "check_cinefocus_available",
    "INSTALL_HINT",
    "CINEFOCUS_PRESETS",
    "CineFocusResult",
    "_load_depth_backend",
    "preview",
    "render",
]
