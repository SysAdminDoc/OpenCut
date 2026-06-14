"""
OpenCut SAM 3 Video Object Segmentation

Text-prompted object segmentation and tracking. SAM 3 extends SAM 2.1 with
native concept-level text prompts — segments and tracks objects by description
("the watermark in the top-right corner") without the click-and-track friction
of SAM 2's point/box prompts.

Licence: SAM License (commercial-permissive)
Repository: https://github.com/facebookresearch/sam3
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
    "pip install sam3 torch  # SAM License (commercial-permissive), "
    "checkpoints auto-download from Hugging Face"
)

SAM3_MODELS = {
    "tiny": "SAM3-Tiny — fastest, text+click+box prompts",
    "small": "SAM3-Small — recommended default",
    "base_plus": "SAM3-Base+ — balanced",
    "large": "SAM3-Large — highest quality",
}

SAM3_PROMPT_TYPES = ["text", "click", "box", "mask"]

SAM3_OUTPUT_FORMATS = ["alpha_video", "matted_video", "mask_frames", "coco_json"]


@dataclass
class SAM3Result:
    output: str = ""
    output_format: str = ""
    frames_processed: int = 0
    objects_tracked: int = 0
    model: str = "small"
    processing_seconds: float = 0.0
    mask_count: int = 0
    engine: str = "sam3"
    text_query: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "output_format", "frames_processed",
                "objects_tracked", "model", "processing_seconds",
                "mask_count", "engine", "text_query", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_sam3_available() -> bool:
    """Return True when SAM3 is importable."""
    return _try_import("sam3") is not None


def segment_video(
    video_path: str,
    prompts: List[Dict],
    text_query: str = "",
    model: str = "small",
    output_format: str = "alpha_video",
    propagate: bool = True,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> SAM3Result:
    """Segment objects in video using SAM 3 with text or spatial prompts.

    Args:
        video_path: Path to source video.
        prompts: List of prompt dicts. Each contains:
            - type: "text" | "click" | "box" | "mask"
            - For text: {"query": str} (e.g., "the watermark")
            - For click: {"x": float, "y": float, "label": 1 (fg) or 0 (bg), "frame": int}
            - For box: {"x1": float, "y1": float, "x2": float, "y2": float, "frame": int}
            - For mask: {"mask_path": str, "frame": int}
        text_query: Convenience text prompt (creates a text-type prompt if prompts is empty).
        model: SAM3 model size — tiny, small, base_plus, large.
        output_format: alpha_video, matted_video, mask_frames, coco_json.
        propagate: Propagate mask across all frames (True) or single-frame only.
        output_path: Output path. Auto-generated if empty.
        on_progress: Optional callback(pct, msg).

    Returns:
        SAM3Result with output path and metadata.
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if text_query and not prompts:
        prompts = [{"type": "text", "query": text_query}]

    if not prompts:
        raise ValueError("At least one prompt is required (text query or click/box)")

    if model not in SAM3_MODELS:
        model = "small"
    if output_format not in SAM3_OUTPUT_FORMATS:
        output_format = "alpha_video"

    if not check_sam3_available():
        raise RuntimeError(f"SAM3 not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, f"Loading SAM3 ({model})...")

    notes: List[str] = []
    resolved_query = text_query or next(
        (p.get("query", "") for p in prompts if p.get("type") == "text"), ""
    )

    if not output_path:
        ext = ".mp4" if "video" in output_format else ".json"
        fd, output_path = tempfile.mkstemp(suffix=ext, prefix="opencut_sam3_")
        os.close(fd)

    try:
        import torch
        from sam3.build_sam import build_sam3_video_predictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = f"sam3_hiera_{model}"
        predictor = build_sam3_video_predictor(model_cfg, device=device)

        if on_progress:
            on_progress(20, "Processing prompts...")

        start_time = time.monotonic()

        state = predictor.init_state(video_path=video_path)

        for prompt in prompts:
            prompt_type = prompt.get("type", "text")
            frame_idx = int(prompt.get("frame", 0))
            obj_id = prompt.get("obj_id", 0)

            if prompt_type == "text":
                predictor.add_new_text(
                    state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    text=prompt.get("query", resolved_query),
                )
            elif prompt_type == "click":
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=[[prompt.get("x", 0), prompt.get("y", 0)]],
                    labels=[prompt.get("label", 1)],
                )
            elif prompt_type == "box":
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=[prompt.get("x1", 0), prompt.get("y1", 0),
                         prompt.get("x2", 0), prompt.get("y2", 0)],
                )

        notes.append(f"Model: {model}")
        notes.append(f"Prompts: {len(prompts)}")
        if resolved_query:
            notes.append(f"Text query: {resolved_query}")

        if on_progress:
            on_progress(40, "Propagating masks...")

        masks = {}
        if propagate:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                masks[frame_idx] = (mask_logits > 0.0).cpu().numpy()

        proc_time = time.monotonic() - start_time
        notes.append(f"Processed in {proc_time:.1f}s")
        notes.append(f"Frames with masks: {len(masks)}")

        if output_format in ("alpha_video", "matted_video"):
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            frame_count = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count in masks:
                        mask = masks[frame_count]
                        if mask.ndim > 2:
                            mask = mask[0]
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if output_format == "alpha_video":
                            frame[mask_resized == 0] = 0
                        else:
                            frame[mask_resized == 0] = [0, 255, 0]
                    writer.write(frame)
                    frame_count += 1
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

        return SAM3Result(
            output=output_path,
            output_format=output_format,
            frames_processed=frame_count if "frame_count" in dir() else len(masks),
            objects_tracked=len(prompts),
            model=model,
            processing_seconds=round(proc_time, 2),
            mask_count=len(masks),
            engine="sam3",
            text_query=resolved_query,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(f"SAM3 import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"SAM3 segmentation failed: {exc}") from exc


def segment_video_auto(
    video_path: str,
    prompts: List[Dict],
    text_query: str = "",
    model: str = "small",
    output_format: str = "alpha_video",
    propagate: bool = True,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> SAM3Result:
    """Segment with SAM3, falling back to SAM2 if SAM3 is unavailable.

    Same interface as segment_video(). When SAM3 is not installed and the
    prompts contain text-type entries, those are converted to a CLIP-based
    region search via text_segment + SAM2 refinement.
    """
    if check_sam3_available():
        return segment_video(
            video_path=video_path,
            prompts=prompts,
            text_query=text_query,
            model=model,
            output_format=output_format,
            propagate=propagate,
            output_path=output_path,
            on_progress=on_progress,
        )

    from opencut.core.segment_sam2 import check_sam2_available as _check_sam2

    if not _check_sam2():
        raise RuntimeError(
            "Neither SAM3 nor SAM2 is installed.\n"
            f"SAM3: {INSTALL_HINT}\n"
            f"SAM2: git clone https://github.com/facebookresearch/sam2 && pip install -e sam2"
        )

    has_text = any(p.get("type") == "text" for p in prompts) or bool(text_query)

    if has_text:
        from opencut.core.text_segment import segment_by_text

        query = text_query or next(
            (p.get("query", "") for p in prompts if p.get("type") == "text"), ""
        )
        if on_progress:
            on_progress(5, f"SAM3 unavailable — falling back to CLIP+SAM2 for '{query}'")

        result = segment_by_text(
            video_path=video_path,
            query=query,
            use_sam2=True,
            sam2_model=model if model in ("tiny", "small", "base_plus", "large") else "small",
            on_progress=on_progress,
        )
        return SAM3Result(
            output=result.get("output_path", result.get("output", "")),
            output_format="mask_frames",
            frames_processed=result.get("frame_count", 0),
            objects_tracked=1,
            model=model,
            processing_seconds=0.0,
            mask_count=result.get("frame_count", 0),
            engine="sam2_fallback",
            text_query=query,
            notes=["Fell back to CLIP+SAM2 (SAM3 not installed)"],
        )

    from opencut.core.segment_sam2 import segment_video as _sam2_segment

    spatial_prompts = [p for p in prompts if p.get("type") in ("click", "box", "mask")]
    if not spatial_prompts:
        raise ValueError("No valid prompts for SAM2 fallback (text prompts require SAM3)")

    if on_progress:
        on_progress(5, "SAM3 unavailable — falling back to SAM2")

    sam2_result = _sam2_segment(
        video_path=video_path,
        prompts=spatial_prompts,
        model=model if model in ("tiny", "small", "base_plus", "large") else "small",
        output_format=output_format,
        propagate=propagate,
        output_path=output_path,
        on_progress=on_progress,
    )
    return SAM3Result(
        output=sam2_result.output,
        output_format=sam2_result.output_format,
        frames_processed=sam2_result.frames_processed,
        objects_tracked=sam2_result.objects_tracked,
        model=sam2_result.model,
        processing_seconds=sam2_result.processing_seconds,
        mask_count=sam2_result.mask_count,
        engine="sam2_fallback",
        text_query="",
        notes=sam2_result.notes + ["Fell back to SAM2 (SAM3 not installed)"],
    )


__all__ = [
    "SAM3Result",
    "check_sam3_available",
    "INSTALL_HINT",
    "SAM3_MODELS",
    "SAM3_PROMPT_TYPES",
    "SAM3_OUTPUT_FORMATS",
    "segment_video",
    "segment_video_auto",
]
