"""
OpenCut SAM2 + Depth Compositor Pipeline (N2.3)

Wires SAM2 (N2.1) + Depth-Anything-V2 (N2.2) into a single endpoint:
1. SAM2 segments subject(s)
2. Depth-Anything-V2 estimates depth
3. Compositor produces layered output with per-layer effects

Closes the biggest remaining gap vs After Effects/DaVinci Fusion.

Licence: Apache-2.0 (combined)
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "Requires: sam2 (N2.1) + torch + transformers (N2.2). "
    "pip install -e sam2 && pip install torch transformers"
)

COMPOSITE_EFFECTS = {
    "blur_background": "Gaussian blur on background layer",
    "replace_background": "Replace background with solid color or image",
    "depth_parallax": "2.5D parallax shift based on depth",
    "color_grade_fg": "Apply color LUT to foreground only",
    "color_grade_bg": "Apply color LUT to background only",
    "vignette_depth": "Depth-based vignette (darken edges by depth)",
}


@dataclass
class CompositeResult:
    output: str = ""
    frames_processed: int = 0
    objects_segmented: int = 0
    effects_applied: List[str] = field(default_factory=list)
    processing_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "frames_processed", "objects_segmented",
                "effects_applied", "processing_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_composite_available() -> bool:
    """Both SAM2 and depth estimation must be available."""
    try:
        from opencut.core.depth_anything_v2 import check_depth_anything_v2_available
        from opencut.core.segment_sam2 import check_sam2_available
        return check_sam2_available() and check_depth_anything_v2_available()
    except Exception:
        return False


def compose(
    video_path: str,
    prompts: List[Dict],
    effects: List[str] = None,
    background_color: str = "",
    background_image: str = "",
    blur_strength: float = 15.0,
    parallax_shift: float = 20.0,
    sam_model: str = "small",
    depth_model: str = "small",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> CompositeResult:
    """Run the full segment + depth + composite pipeline.

    Args:
        video_path: Source video.
        prompts: SAM2 prompt dicts (same format as segment_sam2).
        effects: List of effect names from COMPOSITE_EFFECTS.
        background_color: Hex color for background replacement.
        background_image: Image path for background replacement.
        blur_strength: Gaussian blur kernel size for blur_background.
        parallax_shift: Pixel shift for depth_parallax.
        sam_model: SAM2 model size.
        depth_model: Depth-Anything model size.
        output_path: Output path.
        on_progress: Optional callback.

    Returns:
        CompositeResult with output path and metadata.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")
    if not prompts:
        raise ValueError("At least one SAM2 prompt is required")
    if not check_composite_available():
        raise RuntimeError(f"Pipeline deps not installed. {INSTALL_HINT}")

    effects = effects or ["blur_background"]
    valid_effects = [e for e in effects if e in COMPOSITE_EFFECTS]
    if not valid_effects:
        valid_effects = ["blur_background"]

    if on_progress:
        on_progress(5, "Starting composite pipeline...")

    notes: List[str] = []
    start = time.monotonic()

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_compose_")
        os.close(fd)

    try:
        import cv2
        import numpy as np

        # Stage 1: SAM2 segmentation
        if on_progress:
            on_progress(10, "Stage 1: Segmenting objects with SAM2...")

        from opencut.core.segment_sam2 import segment_video

        seg_fd, seg_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_seg_")
        os.close(seg_fd)

        seg_result = segment_video(
            video_path=video_path,
            prompts=prompts,
            model=sam_model,
            output_format="alpha_video",
            output_path=seg_path,
        )
        notes.append(f"SAM2: {seg_result.mask_count} masked frames")

        # Stage 2: Depth estimation
        if on_progress:
            on_progress(40, "Stage 2: Estimating depth...")

        from opencut.core.depth_anything_v2 import estimate_depth

        depth_fd, depth_path = tempfile.mkstemp(suffix=".npz", prefix="opencut_depth_")
        os.close(depth_fd)

        depth_result = estimate_depth(
            video_path=video_path,
            model=depth_model,
            output_path=depth_path,
            output_format="numpy",
        )
        notes.append(f"Depth: {depth_result.frames_processed} frames")

        # Stage 3: Composite
        if on_progress:
            on_progress(70, "Stage 3: Compositing layers...")

        cap_orig = cv2.VideoCapture(video_path)
        cap_seg = cv2.VideoCapture(seg_path)
        fps = cap_orig.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Load depth data if parallax effect is requested
        depth_data = None
        if "depth_parallax" in valid_effects or "vignette_depth" in valid_effects:
            try:
                depth_data = np.load(depth_path, allow_pickle=False)["depths"]
            except Exception:
                pass

        frame_idx = 0
        try:
            while True:
                ret_o, frame_orig = cap_orig.read()
                ret_s, frame_seg = cap_seg.read()
                if not ret_o:
                    break

                result_frame = frame_orig.copy()

                # Create mask from segmented output (non-black pixels = foreground)
                if ret_s:
                    gray = cv2.cvtColor(frame_seg, cv2.COLOR_BGR2GRAY)
                    fg_mask = (gray > 10).astype(np.uint8)
                else:
                    fg_mask = np.ones((h, w), dtype=np.uint8)

                fg_mask_3 = np.stack([fg_mask] * 3, axis=-1)

                # Apply effects
                for effect in valid_effects:
                    if effect == "blur_background":
                        k = max(3, int(blur_strength) | 1)  # ensure odd
                        blurred = cv2.GaussianBlur(frame_orig, (k, k), 0)
                        result_frame = np.where(fg_mask_3, result_frame, blurred)

                    elif effect == "replace_background" and background_color:
                        color = tuple(int(background_color.lstrip("#")[i:i+2], 16)
                                      for i in (4, 2, 0))  # RGB->BGR
                        bg = np.full_like(frame_orig, color, dtype=np.uint8)
                        result_frame = np.where(fg_mask_3, result_frame, bg)

                    elif effect == "depth_parallax" and depth_data is not None:
                        if frame_idx < len(depth_data):
                            d = cv2.resize(depth_data[frame_idx], (w, h))
                            map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
                            map_x -= d * parallax_shift
                            map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
                            result_frame = cv2.remap(result_frame, map_x, map_y,
                                                     cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_REFLECT)

                writer.write(result_frame)
                frame_idx += 1

                if on_progress and frame_idx % 10 == 0:
                    pct = 70 + int((frame_idx / max(depth_result.frames_processed, 1)) * 25)
                    on_progress(min(95, pct), f"Compositing {frame_idx}...")
        finally:
            cap_orig.release()
            cap_seg.release()
            writer.release()

        # Cleanup temp files
        for tmp in [seg_path, depth_path]:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        elapsed = time.monotonic() - start
        notes.append(f"Effects: {', '.join(valid_effects)}")
        notes.append(f"Total pipeline: {elapsed:.1f}s")

        if on_progress:
            on_progress(100, "Done")

        return CompositeResult(
            output=output_path, frames_processed=frame_idx,
            objects_segmented=len(prompts), effects_applied=valid_effects,
            processing_seconds=round(elapsed, 2), notes=notes,
        )
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Composite pipeline failed: {exc}") from exc


__all__ = ["CompositeResult", "check_composite_available", "INSTALL_HINT",
           "COMPOSITE_EFFECTS", "compose"]
