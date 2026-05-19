"""
OpenCut Depth-Anything-V2 Depth Estimation (N2.2)

Monocular depth estimation: per-pixel depth maps from single frames
or video. 4 sizes (Small 24M to Large 335M), GPU or CPU.

Enables: parallax 2.5D effect, smart reframe, CineFocus depth engine,
depth-guided compositing.

Licence: Apache-2.0
Repository: https://github.com/DepthAnything/Depth-Anything-V2
Paper: NeurIPS 2024
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install depth-anything-v2  # Apache-2.0, monocular depth estimation"

DA2_MODELS = {
    "small": "Small (24M) — fastest, CPU-capable, good for previews",
    "base": "Base (97M) — balanced speed/quality",
    "large": "Large (335M) — highest quality, GPU recommended",
}


@dataclass
class DepthResult:
    output: str = ""
    depth_map_path: str = ""
    frames_processed: int = 0
    model: str = "small"
    processing_seconds: float = 0.0
    min_depth: float = 0.0
    max_depth: float = 1.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "depth_map_path", "frames_processed", "model",
                "processing_seconds", "min_depth", "max_depth", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_depth_anything_v2_available() -> bool:
    return (_try_import("torch") is not None
            and _try_import("transformers") is not None)


def estimate_depth(
    video_path: str,
    model: str = "small",
    output_path: str = "",
    output_format: str = "video",
    on_progress: Optional[Callable] = None,
) -> DepthResult:
    """Estimate per-frame depth maps from video.

    Args:
        video_path: Source video/image path.
        model: small, base, or large.
        output_path: Output path. Auto-generated if empty.
        output_format: "video" (grayscale depth video) or "numpy" (npz).
        on_progress: Optional callback.

    Returns:
        DepthResult with depth map output.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"File not found: {video_path}")
    if not check_depth_anything_v2_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")
    if model not in DA2_MODELS:
        model = "small"

    if on_progress:
        on_progress(5, f"Loading Depth-Anything-V2 ({model})...")

    notes: List[str] = []
    if not output_path:
        suffix = ".mp4" if output_format == "video" else ".npz"
        fd, output_path = tempfile.mkstemp(suffix=suffix, prefix="opencut_depth_")
        os.close(fd)

    try:
        import cv2
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"depth-anything/Depth-Anything-V2-{model.capitalize()}-hf"
        processor = AutoImageProcessor.from_pretrained(model_id)
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)

        if on_progress:
            on_progress(15, "Processing frames...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start = time.monotonic()
        writer = None
        if output_format == "video":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        depth_frames = []
        frame_idx = 0
        global_min, global_max = float("inf"), float("-inf")

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                from PIL import Image
                img = Image.fromarray(bgr[..., ::-1])
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = depth_model(**inputs)
                    depth = outputs.predicted_depth.squeeze().cpu().numpy()

                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                global_min = min(global_min, float(depth.min()))
                global_max = max(global_max, float(depth.max()))

                if output_format == "video" and writer:
                    depth_vis = (depth_norm * 255).astype(np.uint8)
                    depth_rgb = cv2.applyColorMap(
                        cv2.resize(depth_vis, (w, h)), cv2.COLORMAP_INFERNO
                    )
                    writer.write(depth_rgb)
                else:
                    depth_frames.append(depth_norm)

                frame_idx += 1
                if on_progress and total > 0 and frame_idx % 5 == 0:
                    pct = 15 + int((frame_idx / total) * 80)
                    on_progress(min(95, pct), f"Frame {frame_idx}/{total}")
        finally:
            cap.release()
            if writer:
                writer.release()

        if output_format == "numpy" and depth_frames:
            np.savez_compressed(output_path, depths=np.array(depth_frames))

        elapsed = time.monotonic() - start
        notes.append(f"Model: {model} ({DA2_MODELS[model].split(' —')[0]})")
        notes.append(f"Processed {frame_idx} frames in {elapsed:.1f}s")

        del depth_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return DepthResult(
            output=output_path, depth_map_path=output_path,
            frames_processed=frame_idx, model=model,
            processing_seconds=round(elapsed, 2),
            min_depth=round(global_min, 4), max_depth=round(global_max, 4),
            notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"Depth import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Depth estimation failed: {exc}") from exc


def generate_parallax(
    video_path: str,
    shift_x: float = 20.0,
    shift_y: float = 0.0,
    model: str = "small",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> DepthResult:
    """Generate 2.5D parallax effect using depth-based layer separation.

    Shifts foreground/background independently based on depth.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"File not found: {video_path}")
    if not check_depth_anything_v2_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, "Loading depth model for parallax...")

    notes: List[str] = [f"Shift: x={shift_x}, y={shift_y}"]

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_parallax_")
        os.close(fd)

    try:
        import cv2
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"depth-anything/Depth-Anything-V2-{model.capitalize()}-hf"
        processor = AutoImageProcessor.from_pretrained(model_id)
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        start = time.monotonic()
        frame_idx = 0

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                from PIL import Image
                img = Image.fromarray(bgr[..., ::-1])
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    depth = depth_model(**inputs).predicted_depth.squeeze().cpu().numpy()

                depth_norm = cv2.resize(
                    (depth - depth.min()) / (depth.max() - depth.min() + 1e-8),
                    (w, h)
                ).astype(np.float32)

                # Shift pixels based on depth: near=large shift, far=small
                map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
                map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
                map_x -= depth_norm * shift_x
                map_y -= depth_norm * shift_y
                parallax = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                writer.write(parallax)
                frame_idx += 1

                if on_progress and total > 0 and frame_idx % 5 == 0:
                    pct = 10 + int((frame_idx / total) * 85)
                    on_progress(min(95, pct), f"Parallax {frame_idx}/{total}")
        finally:
            cap.release()
            writer.release()

        elapsed = time.monotonic() - start
        notes.append(f"Processed {frame_idx} frames in {elapsed:.1f}s")

        del depth_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return DepthResult(
            output=output_path, depth_map_path="",
            frames_processed=frame_idx, model=model,
            processing_seconds=round(elapsed, 2), notes=notes,
        )
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Parallax generation failed: {exc}") from exc


__all__ = ["DepthResult", "check_depth_anything_v2_available", "INSTALL_HINT",
           "DA2_MODELS", "estimate_depth", "generate_parallax"]
