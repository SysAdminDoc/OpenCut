"""
OpenCut Qwen2.5-VL Smart Timeline Analysis (N3.2)

Vision-language model for video understanding: scene descriptions,
text detection, product identification, quality ratings.

Enables: natural-language clip search, auto-chapter generation,
AI-assisted metadata extraction.

Licence: Apache-2.0
Repository: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "pip install transformers qwen_vl_utils  # Apache-2.0\n"
    "Qwen2.5-VL-7B weights (~16 GB) download on first use."
)

VL_MODELS = {
    "Qwen2.5-VL-3B": "3B — low-spec machines, faster, less detailed",
    "Qwen2.5-VL-7B": "7B — recommended, good detail/speed balance",
}

ANALYSIS_TYPES = [
    "describe_scenes",
    "detect_text",
    "identify_products",
    "rate_quality",
    "generate_chapters",
    "custom_query",
]


@dataclass
class VLAnalysisResult:
    query: str = ""
    response: str = ""
    structured_data: List[Dict] = field(default_factory=list)
    model: str = "Qwen2.5-VL-7B"
    processing_seconds: float = 0.0
    frames_analyzed: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("query", "response", "structured_data", "model",
                "processing_seconds", "frames_analyzed", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_qwen_vl_available() -> bool:
    return (_try_import("transformers") is not None
            and _try_import("torch") is not None)


def _build_query(analysis_type: str, custom_query: str = "") -> str:
    """Build the VL prompt for a given analysis type."""
    prompts = {
        "describe_scenes": (
            "Describe each distinct scene in this video with timestamps. "
            "For each scene, provide: start_time, end_time, description, "
            "mood, and key objects visible."
        ),
        "detect_text": (
            "Identify all text visible in this video. For each text element: "
            "timestamp, content, location in frame, and language."
        ),
        "identify_products": (
            "List all products, brands, or commercial items visible in this video. "
            "For each: timestamp, product name, brand, and location in frame."
        ),
        "rate_quality": (
            "Rate the visual quality of this video on a scale of 1-10. "
            "Evaluate: resolution, lighting, composition, focus, and stability. "
            "Provide per-scene ratings."
        ),
        "generate_chapters": (
            "Identify major topic changes in this video and suggest chapter titles. "
            "Return: chapter_title, start_time, description for each."
        ),
    }
    if analysis_type == "custom_query":
        return custom_query or "Describe what happens in this video."
    return prompts.get(analysis_type, prompts["describe_scenes"])


def analyze_video(
    video_path: str,
    analysis_type: str = "describe_scenes",
    custom_query: str = "",
    model: str = "Qwen2.5-VL-7B",
    max_frames: int = 16,
    on_progress: Optional[Callable] = None,
) -> VLAnalysisResult:
    """Analyze video content using Qwen2.5-VL.

    Args:
        video_path: Path to video file.
        analysis_type: Type of analysis to perform.
        custom_query: Custom question (when type is custom_query).
        model: VL model size.
        max_frames: Maximum frames to sample for analysis.
        on_progress: Optional callback.

    Returns:
        VLAnalysisResult with response and structured data.
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")
    if not check_qwen_vl_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")
    if analysis_type not in ANALYSIS_TYPES:
        analysis_type = "describe_scenes"
    if model not in VL_MODELS:
        model = "Qwen2.5-VL-7B"

    if on_progress:
        on_progress(5, f"Loading {model}...")

    notes: List[str] = []
    query = _build_query(analysis_type, custom_query)

    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"Qwen/{model}-Instruct"

        processor = AutoProcessor.from_pretrained(model_id)
        vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(device)

        if on_progress:
            on_progress(30, "Sampling video frames...")

        # Sample frames from video
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        interval = max(1, total_frames // max_frames)
        frames = []
        frame_times = []
        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, bgr = cap.read()
            if ret:
                from PIL import Image
                img = Image.fromarray(bgr[..., ::-1])
                frames.append(img)
                frame_times.append(round(i / fps, 2))
            if len(frames) >= max_frames:
                break
        cap.release()

        notes.append(f"Sampled {len(frames)} frames")

        if on_progress:
            on_progress(50, "Running VL analysis...")

        start = time.monotonic()

        # Build messages with video frames
        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": query})

        messages = [{"role": "user", "content": content}]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input], images=frames,
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            output_ids = vl_model.generate(**inputs, max_new_tokens=2048)
        response_text = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        elapsed = time.monotonic() - start
        notes.append(f"Analysis: {analysis_type}")
        notes.append(f"Model: {model}")
        notes.append(f"Processed in {elapsed:.1f}s")

        # Try to parse structured data from response
        structured = []
        try:
            import json
            import re
            json_match = re.search(r"\[[\s\S]*?\]", response_text)
            if json_match:
                structured = json.loads(json_match.group())
        except Exception:
            pass

        del vl_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return VLAnalysisResult(
            query=query, response=response_text.strip(),
            structured_data=structured, model=model,
            processing_seconds=round(elapsed, 2),
            frames_analyzed=len(frames), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"VL import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        raise RuntimeError(f"VL analysis failed: {exc}") from exc


__all__ = ["VLAnalysisResult", "check_qwen_vl_available", "INSTALL_HINT",
           "VL_MODELS", "ANALYSIS_TYPES", "analyze_video"]
