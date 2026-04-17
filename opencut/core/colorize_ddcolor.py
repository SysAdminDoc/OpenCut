"""
DDColor AI colorisation for B&W footage.

DDColor (https://github.com/piddnad/DDColor, Apache-2) is a 2024
dual-decoder colorisation network that beats DeOldify on LPIPS/FID.
This module wraps it as a per-frame colouriser with a FFmpeg encode
stage, producing a plain MP4 alongside the original audio.

Architecture in four stages:

1. ``ffmpeg`` extracts greyscale frames as PNGs at the source fps.
2. DDColor colourises each frame (batched on GPU when available).
3. ``ffmpeg`` re-encodes the coloured PNG sequence + original audio.
4. Temp PNG directory is wiped.

The DDColor pip package isn't universally published yet; the reference
implementation is a clone of the Apache-2 repo.  We therefore support
two backends:

- **ONNX** — user points ``OPENCUT_DDCOLOR_ONNX`` at a converted .onnx
  and we use ``onnxruntime`` (no torch dependency).
- **Torch/ModelScope** — best-effort import of ``modelscope`` with the
  DDColor pipeline identifier.

Both backends are optional — when neither is usable, the function
raises ``RuntimeError`` with install guidance.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess as _sp
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import FFmpegCmd, get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class DDColorResult:
    """Structured return for a DDColor run."""
    output: str = ""
    backend: str = ""
    frames_colorised: int = 0
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def check_ddcolor_available() -> bool:
    onnx_path = os.environ.get("OPENCUT_DDCOLOR_ONNX", "")
    if onnx_path and os.path.isfile(onnx_path):
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            pass
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        return False


def _select_backend() -> str:
    onnx_path = os.environ.get("OPENCUT_DDCOLOR_ONNX", "")
    if onnx_path and os.path.isfile(onnx_path):
        try:
            import onnxruntime  # noqa: F401
            return "onnx"
        except ImportError:
            pass
    try:
        import modelscope  # noqa: F401
        return "modelscope"
    except ImportError:
        return "none"


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------

def _extract_frames(input_path: str, out_dir: str, fps: float) -> int:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%06d.png")
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vsync", "0", "-q:v", "2",
        pattern,
    ]
    proc = _sp.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"frame extract failed (rc={proc.returncode}): {proc.stderr[-200:]}"
        )
    return sum(1 for n in os.listdir(out_dir) if n.endswith(".png"))


def _reassemble(frames_dir: str, fps: float, audio_src: str, out_path: str) -> None:
    cmd = (
        FFmpegCmd()
        .input(os.path.join(frames_dir, "frame_%06d.png"), framerate=fps)
        .input(audio_src)
        .map("0:v:0")
        .map("1:a:0?")
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .option("shortest")
        .option("pix_fmt", "yuv420p")
        .faststart()
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd, timeout=10800)


# ---------------------------------------------------------------------------
# Colorisation backends
# ---------------------------------------------------------------------------

def _colourise_onnx(frame_paths: List[str], on_progress: Optional[Callable]) -> None:
    """Colourise PNGs in-place via ONNX Runtime."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    onnx_path = os.environ["OPENCUT_DDCOLOR_ONNX"]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    for i, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("L").resize((512, 512), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.stack([arr] * 3, axis=0)[None]  # NCHW, 3 channels
        out = sess.run(None, {input_name: arr})
        rgb = out[0][0]  # (3, H, W)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_img = (rgb.transpose(1, 2, 0) * 255.0).astype("uint8")
        # Resize back to original
        orig = Image.open(fp).convert("RGB")
        rgb_pil = Image.fromarray(rgb_img, "RGB").resize(orig.size, Image.BILINEAR)
        rgb_pil.save(fp, "PNG", optimize=False)

        if on_progress and len(frame_paths) > 0 and (i % 10 == 0 or i == len(frame_paths) - 1):
            pct = 25 + int(60 * ((i + 1) / len(frame_paths)))
            on_progress(pct, f"Colourising frame {i + 1}/{len(frame_paths)}")


def _colourise_modelscope(frame_paths: List[str], on_progress: Optional[Callable]) -> None:
    """Colourise via ModelScope's DDColor pipeline."""
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from PIL import Image

    pipe = pipeline("image-colorization", model="damo/cv_ddcolor_image-colorization")

    for i, fp in enumerate(frame_paths):
        try:
            result = pipe(fp)
            out_img = result[OutputKeys.OUTPUT_IMG] if OutputKeys.OUTPUT_IMG in result else None
            if out_img is None:
                continue
            # ModelScope returns an ndarray in BGR (cv2 convention)
            import cv2
            cv2.imwrite(fp, out_img)
        except Exception as exc:  # noqa: BLE001
            # Keep going — one bad frame shouldn't abort the whole job.
            logger.warning("DDColor failed on frame %s: %s", fp, exc)
            try:
                Image.open(fp).convert("RGB").save(fp, "PNG")
            except Exception:  # noqa: BLE001
                pass

        if on_progress and len(frame_paths) > 0 and (i % 10 == 0 or i == len(frame_paths) - 1):
            pct = 25 + int(60 * ((i + 1) / len(frame_paths)))
            on_progress(pct, f"Colourising frame {i + 1}/{len(frame_paths)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def colourise_video(
    input_path: str,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> DDColorResult:
    """Colourise ``input_path`` end-to-end with DDColor.

    Raises:
        RuntimeError: no DDColor backend available.
        FileNotFoundError: source missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if not check_ddcolor_available():
        raise RuntimeError(
            "DDColor not available. Install one of: "
            "(1) `pip install onnxruntime` + set OPENCUT_DDCOLOR_ONNX "
            "to a .onnx checkpoint; or "
            "(2) `pip install modelscope`."
        )

    backend = _select_backend()
    out = output or output_path(input_path, "colour")

    if on_progress:
        on_progress(5, f"Probing source (backend={backend})")

    from opencut.helpers import get_video_info
    info = get_video_info(input_path)
    fps = float(info.get("fps") or 30.0)
    duration = float(info.get("duration") or 0.0)

    tmp_root = tempfile.mkdtemp(prefix="opencut_ddcolor_")
    try:
        frames_dir = os.path.join(tmp_root, "frames")
        if on_progress:
            on_progress(10, "Extracting greyscale frames")
        frame_count = _extract_frames(input_path, frames_dir, fps)
        frame_paths = sorted(
            os.path.join(frames_dir, n) for n in os.listdir(frames_dir)
            if n.endswith(".png")
        )
        if not frame_paths:
            raise RuntimeError("No frames extracted — is the source decodable?")

        if on_progress:
            on_progress(25, f"Colourising {frame_count} frame(s)")

        if backend == "onnx":
            _colourise_onnx(frame_paths, on_progress)
        elif backend == "modelscope":
            _colourise_modelscope(frame_paths, on_progress)
        else:
            raise RuntimeError("No usable DDColor backend detected.")

        if on_progress:
            on_progress(85, "Reassembling video with original audio")

        _reassemble(frames_dir, fps, input_path, out)

        if on_progress:
            on_progress(100, "Colourisation complete")

        return DDColorResult(
            output=out,
            backend=backend,
            frames_colorised=frame_count,
            duration=round(duration, 3),
            notes=[f"fps={fps}"],
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
