"""
OpenCut Slate Identifier v1.28.0 — Tier 3 (partial)

Florence-2 VLM reads clapperboard from clip-head frames.
"""
from __future__ import annotations

import io
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List

from opencut.helpers import _try_import, get_ffmpeg_path

logger = logging.getLogger("opencut")
INSTALL_HINT = "pip install transformers torch (Florence-2 already used in OpenCut)"

_florence_cache: Dict[str, object] = {}


def check_slate_id_available() -> bool:
    return _try_import("transformers") is not None


@dataclass
class SlateInfo:
    scene: str = ""
    take: str = ""
    camera: str = ""
    roll: str = ""
    fps: str = ""
    date: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("scene", "take", "camera", "roll", "fps", "date", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def identify(video_path: str, max_head_frames: int = 60) -> SlateInfo:
    """Use Florence-2 to read clapperboard info from clip-head frames."""
    if not check_slate_id_available():
        return SlateInfo(notes=["transformers/torch not available; slate ID skipped"])

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-i", video_path,
            "-vf", f"select=ite(lt(n\\,{max_head_frames}),1,0)",
            "-vsync", "0", "-vframes", str(min(max_head_frames, 5)),
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ]
        raw = subprocess.run(cmd, capture_output=True, timeout=30).stdout
        if not raw:
            return SlateInfo(notes=["No frames extracted from clip head"])

        model_name = "microsoft/Florence-2-base"
        if "processor" not in _florence_cache:
            _florence_cache["processor"] = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            _florence_cache["model"] = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
        processor = _florence_cache["processor"]
        model = _florence_cache["model"]

        from PIL import Image  # type: ignore
        try:
            img = Image.open(io.BytesIO(raw[:65536])).convert("RGB")
        except Exception:
            return SlateInfo(notes=["Could not decode frame for Florence-2"])

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        import torch
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=200,
            )
        caption = processor.batch_decode(generated, skip_special_tokens=True)[0]

        info = SlateInfo(notes=[f"Florence-2 caption: {caption[:100]}"])
        scene_m = re.search(r"scene\s*[#:]?\s*(\w+)", caption, re.IGNORECASE)
        take_m = re.search(r"take\s*[#:]?\s*(\w+)", caption, re.IGNORECASE)
        camera_m = re.search(r"cam(?:era)?\s*[#:]?\s*(\w+)", caption, re.IGNORECASE)
        if scene_m:
            info.scene = scene_m.group(1)
        if take_m:
            info.take = take_m.group(1)
        if camera_m:
            info.camera = camera_m.group(1)
        return info

    except Exception as e:
        logger.warning("Slate ID failed: %s", e)
        return SlateInfo(notes=[f"Slate ID error: {e}"])


__all__ = ["check_slate_id_available", "INSTALL_HINT", "SlateInfo", "identify"]
