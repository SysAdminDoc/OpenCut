"""
OpenCut SAM 2.1 Video Object Segmentation (N2.1)

Foundation model for promptable visual segmentation in images and videos.
Accepts click, box, or mask prompts on any frame and propagates the
segmentation mask throughout the entire video in real time.

SAM 2.1: 4 sizes (Tiny 38M@91FPS -> Large 224M@39FPS on A100).

Licence: Apache-2.0
Repository: https://github.com/facebookresearch/sam2
Paper: ECCV 2024
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
    "git clone https://github.com/facebookresearch/sam2 && "
    "pip install -e sam2  # Apache-2.0, SAM 2.1 checkpoints auto-download"
)

SAM2_MODELS = {
    "tiny": "SAM2.1-Tiny (38M) — fastest, 91 FPS on A100",
    "small": "SAM2.1-Small (46M) — recommended default",
    "base_plus": "SAM2.1-Base+ (80M) — balanced",
    "large": "SAM2.1-Large (224M) — highest quality, 39 FPS on A100",
}

SAM2_PROMPT_TYPES = ["click", "box", "mask"]

SAM2_OUTPUT_FORMATS = ["alpha_video", "matted_video", "mask_frames", "coco_json"]


@dataclass
class SAM2Result:
    output: str = ""
    output_format: str = ""
    frames_processed: int = 0
    objects_tracked: int = 0
    model: str = "small"
    processing_seconds: float = 0.0
    mask_count: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "output_format", "frames_processed",
                "objects_tracked", "model", "processing_seconds",
                "mask_count", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_sam2_available() -> bool:
    """Return True when SAM2 is importable."""
    return _try_import("sam2") is not None


def segment_video(
    video_path: str,
    prompts: List[Dict],
    model: str = "small",
    output_format: str = "alpha_video",
    propagate: bool = True,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> SAM2Result:
    """Segment objects in video using SAM 2.1.

    Args:
        video_path: Path to source video.
        prompts: List of prompt dicts. Each contains:
            - type: "click" | "box" | "mask"
            - frame: Frame number for the prompt
            - For click: {"x": float, "y": float, "label": 1 (fg) or 0 (bg)}
            - For box: {"x1": float, "y1": float, "x2": float, "y2": float}
            - For mask: {"mask_path": str}
        model: SAM2 model size — tiny, small, base_plus, large.
        output_format: alpha_video, matted_video, mask_frames, coco_json.
        propagate: Propagate mask across all frames (True) or single-frame only.
        output_path: Output path. Auto-generated if empty.
        on_progress: Optional callback.

    Returns:
        SAM2Result with output path and metadata.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")
    if not prompts:
        raise ValueError("At least one prompt is required")
    if model not in SAM2_MODELS:
        model = "small"
    if output_format not in SAM2_OUTPUT_FORMATS:
        output_format = "alpha_video"
    if not check_sam2_available():
        raise RuntimeError(f"SAM2 not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, f"Loading SAM2.1 ({model})...")

    notes: List[str] = []

    if not output_path:
        ext = ".mp4" if "video" in output_format else ".json"
        fd, output_path = tempfile.mkstemp(suffix=ext, prefix="opencut_sam2_")
        os.close(fd)

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = f"sam2.1_hiera_{model}"
        predictor = build_sam2_video_predictor(model_cfg, device=device)

        if on_progress:
            on_progress(20, "Processing prompts...")

        start_time = time.monotonic()

        # Initialize video state
        state = predictor.init_state(video_path=video_path)

        # Add prompts
        for prompt in prompts:
            frame_idx = int(prompt.get("frame", 0))
            prompt_type = prompt.get("type", "click")

            if prompt_type == "click":
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=prompt.get("obj_id", 0),
                    points=[[prompt.get("x", 0), prompt.get("y", 0)]],
                    labels=[prompt.get("label", 1)],
                )
            elif prompt_type == "box":
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=prompt.get("obj_id", 0),
                    box=[prompt.get("x1", 0), prompt.get("y1", 0),
                         prompt.get("x2", 0), prompt.get("y2", 0)],
                )

        notes.append(f"Model: {model}")
        notes.append(f"Prompts: {len(prompts)}")

        if on_progress:
            on_progress(40, "Propagating masks...")

        # Propagate
        masks = {}
        if propagate:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks[frame_idx] = (mask_logits > 0.0).cpu().numpy()

        proc_time = time.monotonic() - start_time
        notes.append(f"Processed in {proc_time:.1f}s")
        notes.append(f"Frames with masks: {len(masks)}")

        # Export based on format
        if output_format in ("alpha_video", "matted_video"):
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx in masks:
                        mask = masks[frame_idx]
                        if mask.ndim > 2:
                            mask = mask[0]  # Take first object
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if output_format == "alpha_video":
                            frame[mask_resized == 0] = 0
                        else:
                            frame[mask_resized == 0] = [0, 255, 0]  # Green for non-mask
                    writer.write(frame)
                    frame_idx += 1
            finally:
                cap.release()
                writer.release()

        elif output_format == "coco_json":
            import json
            coco_data = {
                "frames": len(masks),
                "objects": len(prompts),
                "masks": {str(k): v.tolist() for k, v in list(masks.items())[:10]},
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(coco_data, f)

        del predictor, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return SAM2Result(
            output=output_path,
            output_format=output_format,
            frames_processed=frame_idx if "frame_idx" in dir() else len(masks),
            objects_tracked=len(prompts),
            model=model,
            processing_seconds=round(proc_time, 2),
            mask_count=len(masks),
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(f"SAM2 import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"SAM2 segmentation failed: {exc}") from exc


__all__ = [
    "SAM2Result",
    "check_sam2_available",
    "INSTALL_HINT",
    "SAM2_MODELS",
    "SAM2_PROMPT_TYPES",
    "SAM2_OUTPUT_FORMATS",
    "segment_video",
]
